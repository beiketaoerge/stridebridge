"""H1 policy adapters extracted from the latest 0927 controller.

The inference and coordinated assist arithmetic are preserved. The simulation
owner supplies configuration and options without initializing SDK/DDS.
See docs/h1_source_manifest.json for source hashes.
"""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from . import geometry as base
from .geometry import tilt
from .switcher import PolicyAdapter, PolicyCommand
from .left_leg_assist import (settings_from_options, validate_assist_settings,
                             validate_assist_contract, assist_joint_targets)

class Dwaq0816Adapter(PolicyAdapter):
    """h1_deploy_0823 70 x 5 history walking policy, joystick-driven.

    Observation contract matches deploy_real_h1_dwaq.py exactly (adapter-level
    obs/target construction verified numerically against the real deploy
    scripts by check_contract_alignment.py; the 0927-side adapter diff vs the
    sim adapter is <=3.8e-6 under real-model comparison):
    [ang_vel*scale, gravity, cmd, (q-default)*scale, dq*scale, clip(action),
     [sin_l, sin_r, cos_l, cos_r]] with the gait counter incremented AFTER the
    step (first tick phase 0)."""

    def __init__(self, config_path):
        super().__init__("stand", 20)
        self.cfg = yaml.safe_load(Path(config_path).read_text())
        root = Path(config_path).resolve().parents[3]
        self.path = Path(self.cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(root)))
        self.policy = torch.jit.load(str(self.path), map_location="cpu").eval()
        self.ids = self.cfg["joint2motor_idx"]
        self.legs = [7, 3, 4, 5, 10, 8, 0, 1, 2, 11]
        self.default = np.asarray(self.cfg["default_angles"], dtype=np.float32)
        self.max_cmd = np.asarray(self.cfg["max_cmd"], dtype=np.float32)
        self.kp = np.zeros(20, dtype=np.float32)
        self.kd = self.kp.copy()
        self.kp[self.ids] = self.cfg["kps"]
        self.kd[self.ids] = self.cfg["kds"]
        self.action = np.zeros(19, dtype=np.float32)
        self.obs = np.zeros(self.cfg["num_obs"], dtype=np.float32)
        self.history = np.zeros((self.cfg["history_length"], self.cfg["num_obs"]), dtype=np.float32)
        self.counter = 0
        self.cmd = np.zeros(3, dtype=np.float32)   # joystick values [-1, 1]
        if self.cfg.get("hold_target_on_limit_violation") or self.cfg.get("max_target_delta") is not None:
            raise ValueError("This adapter expects the supplied DWAQ target-clipping configuration")
        # Inference warm-up, as deploy_real_h1_dwaq.py does, so the first live
        # tick is not delayed by lazy init.
        with torch.no_grad():
            for _ in range(20):
                self.policy(torch.zeros(1, self.cfg["history_length"] * self.cfg["num_obs"]))

    def reset(self, state):
        self.action[:] = 0
        self.history[:] = 0
        self.counter = 0
        self.cmd[:] = 0

    def step(self, state, command_scale):
        w, x, y, z = state.extra["quat"]
        terms = [state.extra["omega"] * self.cfg["ang_vel_scale"],
                 np.array([2 * (-z * x + w * y), -2 * (w * x + y * z), 1 - 2 * (w * w + z * z)]),
                 self.cmd * np.asarray(self.cfg["cmd_scale"], dtype=np.float32) * self.max_cmd,
                 (state.q[self.ids] - self.default) * self.cfg["dof_pos_scale"],
                 state.dq[self.ids] * self.cfg["dof_vel_scale"],
                 self.action]
        phase_cfg = self.cfg.get("gait_phase", {})
        if phase_cfg.get("enable", False):
            phase_l = self.counter * self.cfg["control_dt"] / phase_cfg["period"] % 1
            phase_r = (phase_l + phase_cfg["offset"]) % 1
            terms.append(np.array([np.sin(2 * np.pi * phase_l), np.sin(2 * np.pi * phase_r),
                                   np.cos(2 * np.pi * phase_l), np.cos(2 * np.pi * phase_r)]))
        self.obs[:] = np.clip(np.concatenate(terms), -100, 100)
        if self.counter == 0:
            self.history[:] = self.obs
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = self.obs
        with torch.no_grad():
            self.action = np.clip(
                self.policy(torch.from_numpy(self.history.reshape(1, -1))).numpy().reshape(-1),
                -100, 100).astype(np.float32)
        target = np.zeros(20, dtype=np.float32)
        target[self.ids] = np.clip(self.default + self.cfg["action_scale"] * self.action,
                                   self.cfg["joint_limit_lower"], self.cfg["joint_limit_upper"])
        self.counter += 1                      # after the step, per the real contract
        self.last_command = PolicyCommand(target, self.kp.copy(), self.kd.copy())
        return self.last_command

    def ready(self, state, command):
        return tilt(state.extra["quat"]) < 30 and np.max(abs(state.dq[self.legs])) < 8

    def accept_executed_command(self, command):
        self.last_executed = command.q.copy()

class MotionAdapter(PolicyAdapter):
    """0927 model_30000 with the realfix obs-104 contract, owned motion clock.

    Adds the tested smooth position overlay after policy target limiting;
    the native raw-action history and PD gains remain unchanged."""

    def __init__(self, owner):
        super().__init__("cross", 20)
        self.owner = owner
        self.options = owner.options
        self.cfg = owner.config
        self.ids = self.cfg.joint2motor_idx
        self.motion = owner.motion
        self.policy = owner.policy
        self.assist_settings = settings_from_options(self.options)
        validate_assist_settings(self.assist_settings, self.options.motion_clock)
        validate_assist_contract(self.assist_settings, self.cfg, self.motion)
        self.kp = np.zeros(20, dtype=np.float32)
        self.kd = self.kp.copy()
        self.kp[self.ids] = self.cfg.kps
        self.kd[self.ids] = self.cfg.kds
        self.obs = np.zeros(104, dtype=np.float32)
        self.action = np.zeros(19, dtype=np.float32)
        self.frame = 0
        self.started = False
        self.yaw_align = np.array([1, 0, 0, 0], dtype=np.float32)
        self.assist_weight = self.assist_knee_delta = 0.
        self.assist_base_target = np.zeros(19, dtype=np.float32)
        self.assist_target = self.assist_base_target.copy()

    @property
    def motion_len(self):
        return int(self.motion["joint_pos"].shape[0])

    def reset(self, state):
        self.frame = 0
        self.started = False
        self.action[:] = 0

    def step(self, state, command_scale):
        i = min(self.frame, self.motion_len - 1)
        self.sample_frame = i
        quat = state.extra["quat"]
        if not self.started:
            self.yaw_align = base.quat_mul(base.yaw_quat(quat),
                                           base.quat_conj(base.yaw_quat(self.motion["body_quat_w"][0, 0])))
        ref_quat = base.quat_mul(self.yaw_align, self.motion["body_quat_w"][i, 0])
        relative = base.quat_mul(base.quat_conj(quat), ref_quat)
        self.obs[:19] = self.motion["joint_pos"][i]
        self.obs[19:38] = self.motion["joint_vel"][i]
        self.obs[38:44] = base.quat_to_matrix(relative)[:, :2].reshape(-1)
        self.obs[44:47] = state.extra["omega"]
        self.obs[47:66] = state.q[self.ids] - self.cfg.default_angles
        self.obs[66:85] = state.dq[self.ids]
        self.obs[85:104] = self.action
        self.obs[:] = np.clip(self.obs, -self.cfg.clip_observations, self.cfg.clip_observations)
        with torch.no_grad():
            self.raw_action = self.policy(
                torch.from_numpy(self.obs).unsqueeze(0)).numpy().reshape(-1).astype(np.float32)
        self.clipped_action = np.clip(self.raw_action, -self.cfg.clip_actions, self.cfg.clip_actions)
        self.raw_target = self.cfg.default_angles + self.cfg.action_scale * self.clipped_action
        target = np.clip(self.raw_target, self.cfg.joint_limit_lower, self.cfg.joint_limit_upper)
        self.assist_base_target = target.copy()
        target, self.assist_weight, self.assist_knee_delta = assist_joint_targets(
            target, self.cfg.joint_limit_lower, self.cfg.joint_limit_upper,
            i if self.started else 0, self.assist_settings)
        self.assist_target = target.copy()
        q = np.zeros(20, dtype=np.float32)
        q[self.ids] = target
        self.last_command = PolicyCommand(q, self.kp.copy(), self.kd.copy())
        return self.last_command

    def ready(self, state, command):
        if not self.started:
            return (tilt(state.extra["quat"]) < self.options.entry_max_tilt
                    and np.max(abs(state.dq[self.ids[:10]])) < self.options.entry_max_speed)
        return tilt(state.extra["quat"]) < 30 and np.max(abs(state.dq[self.ids[:10]])) < 8

    def accept_executed_command(self, command):
        self.last_executed = command.q.copy()
        weight = command.metadata.get("blend_weight", 0)
        if weight > 0:
            self.started = True
            self.action[:] = self.raw_action
            if self.options.motion_clock == "blended" or weight >= 1 - 1e-6:
                self.frame += 1
        elif not self.started:
            # Idle inference must not consume the 3.1 s clip or recursively
            # feed unexecuted actions into the entry observation.
            self.action[:] = 0
