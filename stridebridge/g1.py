"""Adapters for the pinned Unitree RL Lab G1 29-DoF ONNX policies.

Observation order, per-term history, motor mapping, gains and action conversion
follow Unitree's deploy.yaml and C++ deployment interfaces (Apache-2.0).
Motion time is owned by the switcher; the clip is held during entry blending.
"""
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation, Slerp

from .geometry import gravity, quat_to_matrix, yaw_quat, tilt
from .switcher import PolicyAdapter, PolicyCommand


def torso_rotation(root_quat, q):
    # SDK slots 12, 13, 14 are waist yaw, roll, pitch (intrinsic ZXY).
    return quat_to_matrix(root_quat) @ Rotation.from_euler("ZXY", q[12:15]).as_matrix()


class G1Policy(PolicyAdapter):
    def __init__(self, name, directory):
        super().__init__(name, 29)
        directory = Path(directory)
        self.cfg = yaml.safe_load((directory / "params/deploy.yaml").read_text())
        self.ids = np.asarray(self.cfg["joint_ids_map"], dtype=int)
        if sorted(self.ids.tolist()) != list(range(29)):
            raise ValueError("Expected the complete G1 29-DoF motor mapping")
        self.dt = float(self.cfg["step_dt"])
        self.default = np.asarray(self.cfg["default_joint_pos"], dtype=np.float32)
        action_cfg = self.cfg["actions"]["JointPositionAction"]
        self.offset = np.asarray(action_cfg["offset"], dtype=np.float32)
        self.scale = np.asarray(action_cfg["scale"], dtype=np.float32)
        # Gains in deploy.yaml are already in motor order; poses are policy order.
        self.kp = np.asarray(self.cfg["stiffness"], dtype=np.float32)
        self.kd = np.asarray(self.cfg["damping"], dtype=np.float32)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / "exported/policy.onnx"),
                                           sess_options=options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.action = np.zeros(29, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)  # physical m/s, m/s, rad/s
        self.history = {}
        self.last_obs = None

    def reset(self, state):
        self.action[:] = 0
        self.history.clear()

    def terms(self, state):
        return {"base_ang_vel": state.extra["omega"],
                "projected_gravity": gravity(state.extra["quat"]),
                "velocity_commands": self.command,
                "joint_pos_rel": state.q[self.ids] - self.default,
                "joint_vel_rel": state.dq[self.ids],
                "last_action": self.action}

    def step(self, state, command_scale):
        terms = self.terms(state)
        blocks = []
        for name, cfg in self.cfg["observations"].items():
            value = np.asarray(terms[name], dtype=np.float32)
            if cfg.get("clip") is not None:
                value = np.clip(value, *cfg["clip"])
            value = value * np.asarray(cfg.get("scale", 1.), dtype=np.float32)
            length = int(cfg.get("history_length", 1))
            if name not in self.history:
                self.history[name] = deque([value.copy() for _ in range(length)], maxlen=length)
            else:
                self.history[name].append(value.copy())
            blocks.append(np.concatenate(list(self.history[name])))
        self.last_obs = np.concatenate(blocks).astype(np.float32)[None, :]
        if self.last_obs.shape[1] != self.input.shape[1] or not np.isfinite(self.last_obs).all():
            raise ValueError("Observation does not match the ONNX policy contract")
        self.raw_action = self.session.run(None, {self.input.name: self.last_obs})[0].reshape(-1)
        if self.raw_action.shape != (29,) or not np.isfinite(self.raw_action).all():
            raise ValueError("Invalid G1 policy action")
        q = np.empty(29, dtype=np.float32)
        q[self.ids] = self.offset + self.scale * self.raw_action
        self.action[:] = self.raw_action
        return PolicyCommand(q, self.kp.copy(), self.kd.copy())

    def ready(self, state, command):
        return tilt(state.extra["quat"]) < 30

    def accept_executed_command(self, command):
        # The velocity policy observes the action actually sent to the robot,
        # expressed in its own offset/scale convention while another policy owns it.
        self.action[:] = (command.q[self.ids] - self.offset) / self.scale


class G1Mimic(G1Policy):
    def __init__(self, directory, start=2.8, end=24., fps=60.):
        super().__init__("motion", directory)
        files = list((Path(directory) / "params").glob("*.csv"))
        if len(files) != 1:
            raise ValueError("Expected one reference motion CSV")
        self.motion = np.loadtxt(files[0], delimiter=",")
        if self.motion.ndim != 2 or self.motion.shape[1] != 36 or not np.isfinite(self.motion).all():
            raise ValueError("Expected finite root xyz, xyzw quaternion, and 29 motor positions")
        self.fps = fps
        self.duration = len(self.motion) / fps
        if not 0 <= start < end <= (len(self.motion)-1)/fps:
            raise ValueError("Motion interval is outside the reference clip")
        self.start, self.end = float(start), float(end)
        self.joint_velocity = np.diff(self.motion[:, 7:], axis=0) * fps
        self.joint_velocity = np.vstack([self.joint_velocity, self.joint_velocity[-1]])
        self.rotations = Rotation.from_quat(self.motion[:, 3:7])
        self.slerp = Slerp(np.arange(len(self.motion)) / fps, self.rotations)
        self.time = self.start
        self.started = False
        self.alignment = np.eye(3)

    @property
    def done(self):
        return self.time >= self.end - 1e-8

    def reset(self, state):
        super().reset(state)
        self.time = self.start
        self.started = False

    def reference(self):
        # Match upstream MotionLoader_'s rounded frame + linear extrapolation.
        t = min(self.time, self.end)
        index = min(int(np.floor(t / self.duration * (len(self.motion)-1) + .5)), len(self.motion)-2)
        weight = np.round((t * self.fps - index) * 1e5) / 1e5
        q = self.motion[index, 7:] * (1-weight) + self.motion[index+1, 7:] * weight
        dq = self.joint_velocity[index] * (1-weight) + self.joint_velocity[index+1] * weight
        # Constant angular-velocity interpolation, including the small upstream extrapolation.
        r0, r1 = self.rotations[index], self.rotations[index+1]
        rot = r0 * Rotation.from_rotvec((r0.inv()*r1).as_rotvec() * weight)
        root = rot.as_quat()[[3, 0, 1, 2]]
        return q, dq, root

    def terms(self, state):
        q, dq, root = self.reference()
        real_torso = torso_rotation(state.extra["quat"], state.q)
        if not self.started:
            real_q = Rotation.from_matrix(real_torso).as_quat()[[3, 0, 1, 2]]
            self.alignment = quat_to_matrix(yaw_quat(real_q)) @ quat_to_matrix(yaw_quat(root)).T
        ref_torso = self.alignment @ torso_rotation(root, q)
        terms = super().terms(state)
        terms["motion_command"] = np.concatenate([q[self.ids], dq[self.ids]])
        terms["motion_anchor_ori_b"] = (real_torso.T @ ref_torso)[:, :2].reshape(-1)
        return terms

    def step(self, state, command_scale):
        if not self.started:
            self.action[:] = 0
        result = super().step(state, command_scale)
        if not self.started:
            self.action[:] = 0
        return result

    def accept_executed_command(self, command):
        weight = command.metadata.get("blend_weight", 0.)
        if weight > 0:
            self.started = True
            self.action[:] = self.raw_action
        if weight >= 1-1e-6:
            self.time = min(self.end, self.time+self.dt)

    def ready(self, state, command):
        limit = 30 if self.started else 15
        return tilt(state.extra["quat"]) < limit
