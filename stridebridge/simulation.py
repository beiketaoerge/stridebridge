"""A single MuJoCo command owner, with PD recomputed at each physics step."""
from pathlib import Path
import numpy as np
import mujoco

from .switcher import RobotState
from .geometry import quat_to_matrix


class Simulation:
    def __init__(self, xml, motor_count, joint_names=None, motor_ids=None):
        self.xml = Path(xml)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml))
        self.model.opt.timestep = .002
        self.data = mujoco.MjData(self.model)
        self.motor_count = motor_count
        self.ids = np.arange(motor_count) if motor_ids is None else np.asarray(motor_ids)
        if joint_names is None:
            self.aidx = np.arange(self.model.nu)
            jids = self.model.actuator_trnid[:, 0]
        else:
            jids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names])
            if (jids < 0).any():
                raise ValueError("Robot model is missing a controlled joint")
            self.aidx = np.array([next(a for a in range(self.model.nu)
                                      if self.model.actuator_trnid[a, 0] == j) for j in jids])
        if len(jids) != len(self.ids) or len(set(self.aidx)) != len(self.ids):
            raise ValueError("Actuator mapping is not one-to-one")
        self.qidx = self.model.jnt_qposadr[jids]
        self.vidx = self.model.jnt_dofadr[jids]
        self.lower = np.full(motor_count, -np.inf, dtype=np.float32)
        self.upper = np.full(motor_count, np.inf, dtype=np.float32)
        limited = self.model.jnt_limited[jids].astype(bool)
        self.lower[self.ids[limited]] = self.model.jnt_range[jids[limited], 0]
        self.upper[self.ids[limited]] = self.model.jnt_range[jids[limited], 1]

    def initialize(self, q, root_position=None, root_quat=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qidx] = np.asarray(q)[self.ids]
        if root_position is not None:
            self.data.qpos[:3] = root_position
        if root_quat is not None:
            self.data.qpos[3:7] = root_quat
        mujoco.mj_forward(self.model, self.data)

    def read(self):
        q, dq = np.zeros(self.motor_count, dtype=np.float32), np.zeros(self.motor_count, dtype=np.float32)
        q[self.ids] = self.data.qpos[self.qidx]
        dq[self.ids] = self.data.qvel[self.vidx]
        quat = self.data.qpos[3:7].astype(np.float32).copy()
        # Free-joint rotational velocity is expressed in the local body frame.
        return RobotState(float(self.data.time), q, dq,
                          {"quat": quat, "omega": self.data.qvel[3:6].astype(np.float32).copy(),
                           "velocity": quat_to_matrix(quat).T @ self.data.qvel[:3],
                           "height": float(self.data.qpos[2])})

    def step(self, command, dt=.02):
        command.validate(self.motor_count)
        steps = round(dt / self.model.opt.timestep)
        if abs(steps*self.model.opt.timestep-dt) > 1e-9:
            raise ValueError("Control timestep must be a multiple of physics timestep")
        for _ in range(steps):
            torque = command.kp[self.ids]*(command.q[self.ids]-self.data.qpos[self.qidx]) - command.kd[self.ids]*self.data.qvel[self.vidx]
            limited = self.model.actuator_ctrllimited[self.aidx].astype(bool)
            bounds = self.model.actuator_ctrlrange[self.aidx]
            torque[limited] = np.clip(torque[limited], bounds[limited, 0], bounds[limited, 1])
            self.data.ctrl[:] = 0
            self.data.ctrl[self.aidx] = torque
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite simulated robot state")
