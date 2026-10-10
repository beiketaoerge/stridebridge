#!/usr/bin/env python3
"""Real-robot H1 controller: 0816 dwaq walking <-> 0927 model_30000 crossing via h1switch.

P0 fixes over the 0924 loopback integration (0923/deploy version), per the
2026-10-02 audit workflow:
  1. The `lo --sim-auto-start` entry restriction is removed: any interface and
     DDS domain work; without --sim-auto-start the flow waits for the remote
     [start] key, ramps to the 0816 default pose over 2 s, then hands control
     to the switcher with the 0816 policy standing.
  2. Dwaq0816Adapter reads the joystick ([ly, -lx, -rx] * cmd_scale * max_cmd,
     identical to deploy_real_h1_dwaq.py) so the operator can walk in and out.
     The stick only drives the stand policy while it is solely active.
  3. The crossing switch is operator-triggered (remote [A]) on the real robot;
     the gates (tilt<15 deg, leg speed<2 rad/s, target gap<1 rad stable 0.1 s)
     still apply, and the operator parks the robot at the calibrated 0.41 m
     board gap before pressing [A]. The rig auto-entry after --stand-seconds is
     kept for --sim-prepare loopback runs. The bounded deadline now starts at
     the switch request, so waiting for the operator is not time-limited.
  4. The old 0923 knee/PD assist remains removed. The coordinated trailing-left
     hip/knee/ankle position overlay verified in native DDS sim2sim is enabled
     by default (frames 90..114, knee assist cap 2.02 rad). Disable with
     --left-assist-mode off. No PD-gain or raw-policy-history modification.
  Also: gait-phase counter now increments after the step (matches the 0816 real
  contract, fixing the 0924 off-by-one), the limiter uses the 0927 crossing
  config limits exactly like the validated rig (no config intersection), and
  [select] returns startup to zero torque; during policy control it exits to
  damping.

Crossing contract source: 0927 bundle deploy_real_h1_motion_104_realfix.py
(imported as base; {LEGGED_GYM_ROOT_DIR} resolves to the 0927 bundle root).
Stand contract source: h1_deploy_0823 deploy_real_h1_dwaq.py + its yaml.
Entry blend default 0.4 s / exit 0.4 s (validated with this policy pair in
sim; the 0924 rig used 0.2 s entry with model29999).
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
H1_ASSET_ROOT = HERE.parents[1] / "assets/local/h1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(H1_ASSET_ROOT / "0927/deploy/deploy_real"))

import deploy_real_h1_motion_104_realfix as base
from common.remote_controller import KeyMap
from h1_policy_switcher import (CommandLimiter, PolicyAdapter, PolicyCommand,
                                PolicySwitcher, RobotBackend, RobotState)
from left_leg_assist import (add_assist_arguments, assist_joint_targets,
                             settings_from_options, validate_assist_contract,
                             validate_assist_settings)
from dataclasses import asdict
import left_leg_assist as left_assist_module

OPTIONS = None
UPSTREAM_COMMIT = "66a0da8bf1e4728fb663b8e72e880a8119838260"
DEFAULT_CROSS_CONFIG = ("h1_motion_104_coretestvideo_gmr_stairhome_obstacle0p30_"
                        "yawm76p5_model30000.yaml")


class OperatorStop(Exception):
    """The operator selected a stop before the next position command."""


class StartupCancelled(OperatorStop):
    """Return to zero torque and require a fresh manual startup."""


def tilt(quat):
    return float(np.rad2deg(np.arccos(np.clip(1 - 2 * np.sum(quat[1:3] ** 2), -1, 1))))


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
        self.cfg = owner.config
        self.ids = self.cfg.joint2motor_idx
        self.motion = owner.motion
        self.policy = owner.policy
        self.assist_settings = settings_from_options(OPTIONS)
        validate_assist_settings(self.assist_settings, OPTIONS.motion_clock)
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
            return (tilt(state.extra["quat"]) < OPTIONS.entry_max_tilt
                    and np.max(abs(state.dq[self.ids[:10]])) < OPTIONS.entry_max_speed)
        return tilt(state.extra["quat"]) < 30 and np.max(abs(state.dq[self.ids[:10]])) < 8

    def accept_executed_command(self, command):
        self.last_executed = command.q.copy()
        weight = command.metadata.get("blend_weight", 0)
        if weight > 0:
            self.started = True
            self.action[:] = self.raw_action
            if OPTIONS.motion_clock == "blended" or weight >= 1 - 1e-6:
                self.frame += 1
        elif not self.started:
            # Idle inference must not consume the 3.1 s clip or recursively
            # feed unexecuted actions into the entry observation.
            self.action[:] = 0


class DdsBackend(RobotBackend):
    def __init__(self, owner):
        super().__init__(20, owner.config.control_dt)
        self.owner = owner
        self.last_tick = -1
        self.last_tick_wall = time.monotonic()

    def read_state(self):
        if self.owner.feedback is None:
            raise RuntimeError("No valid LowState feedback received")
        msg, received = self.owner.feedback
        raw_quat = np.asarray(msg.imu_state.quaternion, dtype=np.float64)
        # Validate the RAW IMU quaternion before transform_imu_data, which
        # normalizes and would otherwise hide a garbage input.
        if not np.isfinite(raw_quat).all() or abs(np.linalg.norm(raw_quat) - 1) > 0.05:
            raise RuntimeError(f"Invalid raw IMU quaternion: {np.round(raw_quat, 4).tolist()}")
        # Liveness beyond receive time: a frozen publisher replaying the same
        # hardware tick would pass an age check forever.
        now = time.monotonic()
        if msg.tick == self.last_tick:
            if now - self.last_tick_wall > 0.1:
                raise RuntimeError(f"LowState tick frozen at {msg.tick} for {now - self.last_tick_wall:.2f}s")
        else:
            self.last_tick = msg.tick
            self.last_tick_wall = now
        q = np.array([m.q for m in msg.motor_state], dtype=np.float32)
        dq = np.array([m.dq for m in msg.motor_state], dtype=np.float32)
        torso = self.owner.config.torso_idx
        quat, omega = base.transform_imu_data(q[torso], dq[torso],
                                              np.asarray(msg.imu_state.quaternion),
                                              np.asarray([msg.imu_state.gyroscope]))
        quat = np.asarray(quat, dtype=np.float32)
        omega = np.asarray(omega, dtype=np.float32).reshape(3)
        age = now - received
        if (age > .1 or not np.isfinite(np.r_[q, dq, quat, omega]).all()
                or abs(np.linalg.norm(quat) - 1) > .02 or tilt(quat) > 45):
            raise RuntimeError(f"Invalid handoff state: age={age:.4f}s tilt={tilt(quat):.2f}deg")
        return RobotState(time.monotonic(), q, dq, {"quat": quat, "omega": omega,
                                                    "state_tick": msg.tick, "age": age,
                                                    "received_at": received})

    def write_command(self, command, *, state_timestamp=None):
        check_select = (self.owner._check_startup_select
                        if self.owner.startup_in_progress else self.owner._check_run_select)
        check_select()
        command.validate(20)
        if np.any(command.kp < 0) or np.any(command.kd < 0):
            raise ValueError("PD gains must be nonnegative")
        # A fresh incoming message does not make a command computed from an
        # old snapshot fresh. Reject slow inference as well as lost feedback.
        self.read_state()
        if state_timestamp is not None and time.monotonic() - state_timestamp > 0.1:
            raise RuntimeError("Position command uses feedback older than 0.1 s")
        for i, motor in enumerate(self.owner.low_cmd.motor_cmd):
            motor.q = float(command.q[i])
            motor.dq = 0.0
            motor.kp = float(command.kp[i])
            motor.kd = float(command.kd[i])
            motor.tau = 0.0
        check_select()
        self.owner.send_cmd(self.owner.low_cmd, before_write=check_select)

    def zero_torque(self):
        # Stop commands bypass feedback validation so an invalid state cannot
        # prevent the operator from disabling position control.
        for motor in self.owner.low_cmd.motor_cmd:
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = 0.0
            motor.tau = 0.0
        self.owner.send_cmd(self.owner.low_cmd)

    def damping(self):
        base.create_damping_cmd(self.owner.low_cmd)
        for motor in self.owner.low_cmd.motor_cmd:
            motor.dq = 0.0
        self.owner.send_cmd(self.owner.low_cmd)


class H1SwitchController(base.Controller):
    def __init__(self, *args, run_steps=None, **kwargs):
        self.feedback = None
        self.rejected_states = 0
        self.crossing_start_row = -1
        self.events = []
        self.trace_weights = []
        self.trace_kp = []
        self.trace_kd = []
        self.trace_stand_obs = []
        self.trace_age = []
        self.trace_gap = []
        self.trace_cmd = []
        self.trace_assist_weight = []
        self.trace_assist_knee_delta = []
        self.trace_assist_base_target = []
        self.trace_assist_target = []
        self.startup_in_progress = False
        self._startup_manual_restart = False
        super().__init__(*args, **kwargs)
        self.run_steps = run_steps
        self.sim_prepare = bool(OPTIONS.sim_prepare)
        self.backend = DdsBackend(self)
        self.stand = Dwaq0816Adapter(OPTIONS.stand_config)
        self.cross = MotionAdapter(self)
        ids = self.config.joint2motor_idx
        # Limiter uses the 0927 crossing config limits, exactly like the
        # validated 0924 rig (NOT the two-config intersection: the intersection
        # clamps hip_pitch to +-1.57 while the 0927 contract allows -3.14..2.53).
        lower = np.zeros(20, dtype=np.float32)
        upper = lower.copy()
        lower[ids] = self.config.joint_limit_lower
        upper[ids] = self.config.joint_limit_upper
        # Two limit sets, swapped per tick: while the stand policy is solely
        # active the 0816 contract limits apply (hip_pitch +-1.57, as its own
        # real deploy always clips); during any blend or the crossing the 0927
        # limits apply (the motion needs hip_pitch to -3.14..2.53).
        walk_lower, walk_upper = lower.copy(), upper.copy()
        sids = np.asarray(self.stand.ids)
        walk_lower[sids] = np.maximum(walk_lower[sids],
                                      np.asarray(self.stand.cfg["joint_limit_lower"], dtype=np.float32))
        walk_upper[sids] = np.minimum(walk_upper[sids],
                                      np.asarray(self.stand.cfg["joint_limit_upper"], dtype=np.float32))
        self.lim_cross = (lower, upper)
        self.lim_walk = (walk_lower, walk_upper)
        self.switcher = PolicySwitcher(self.stand, self.cross, "stand",
                                       OPTIONS.entry_seconds, self.config.control_dt,
                                       stable_seconds=OPTIONS.stable_seconds,
                                       max_target_gap=OPTIONS.max_target_gap,
                                       limiter=CommandLimiter(20, lower, upper))
        print(f"h1switch {UPSTREAM_COMMIT}; stand policy {self.stand.path}", flush=True)
        print(f"cross policy {self.config.policy_path}", flush=True)
        print(f"entry {OPTIONS.entry_seconds}s exit {OPTIONS.exit_seconds}s; "
              f"left-leg assist={self.cross.assist_settings.mode} "
              f"(additional knee target cap {self.cross.assist_settings.knee_cap} rad)", flush=True)

    def LowStateGoHandler(self, msg):
        if msg.crc != base.CRC().Crc(msg):
            self.rejected_states += 1
            return
        super().LowStateGoHandler(msg)
        self.feedback = (msg, time.monotonic())

    def LowStateHgHandler(self, msg):
        # Same CRC/feedback path if an hg-type config is ever used; otherwise
        # self.feedback would never update and read_state would crash.
        if msg.crc != base.CRC().Crc(msg):
            self.rejected_states += 1
            return
        super().LowStateHgHandler(msg)
        self.feedback = (msg, time.monotonic())

    def send_cmd(self, cmd, *, before_write=None):
        cmd.crc = base.CRC().Crc(cmd)
        if before_write is not None:
            before_write()
        self.lowcmd_publisher_.Write(cmd)

    def _check_startup_select(self):
        if self.remote_controller.button[KeyMap.select] == 1:
            raise StartupCancelled("Operator [select]: return startup to zero torque")

    def _check_run_select(self):
        if self.remote_controller.button[KeyMap.select] == 1:
            raise OperatorStop("Operator [select]: stop policy control")

    def zero_torque_state(self):
        if self.sim_auto_start and not self._startup_manual_restart:
            self._check_startup_select()
            print("Sim auto-start: skip zero torque wait.", flush=True)
            return
        print("Zero torque. Release [select], then press [start] again to "
              "begin the pose ramp.", flush=True)
        # Observe held Start rather than treating it as a new press when a
        # cancelled ramp returns here. Select always wins over Start.
        prev_start = self.remote_controller.button[KeyMap.start]
        while True:
            self.backend.zero_torque()
            self.backend.read_state()
            start = self.remote_controller.button[KeyMap.start]
            selected = self.remote_controller.button[KeyMap.select] == 1
            if not selected and start == 1 and prev_start == 0:
                return
            prev_start = start
            time.sleep(self.config.control_dt)

    def startup(self):
        """Select cancels startup without exiting or entering damping."""
        self.startup_in_progress = True
        try:
            while True:
                try:
                    self.zero_torque_state()
                    self._check_startup_select()
                    self.move_to_motion_start()
                    self._check_startup_select()
                    self.default_pos_state()
                    self._check_startup_select()
                    self._startup_manual_restart = False
                    return
                except StartupCancelled:
                    self._startup_manual_restart = True
                    self.backend.zero_torque()
                    print("Startup cancelled by [select]; zero torque active. "
                          "Release the keys and press [start] to try again.", flush=True)
        finally:
            self.startup_in_progress = False

    def move_to_motion_start(self):
        """2 s ramp from the measured pose to the 0816 default stance with the
        0816 gains (deploy_real_h1_dwaq.py's move_to_default_pos equivalent).
        On hardware the motors hold this fine; in the MuJoCo loopback rig the
        plain-PD crouch collapses (known limitation, same reason the 0924
        integration skipped it), so the rig path lets the 0816 policy catch
        the robot directly instead."""
        self._check_startup_select()
        if OPTIONS.net == "lo":
            print("Loopback rig: skipping the pose ramp; the 0816 stand policy "
                  "takes over directly (0924-validated path).", flush=True)
            return
        dt = self.config.control_dt
        state = self.backend.read_state()
        start_q = state.q.copy()
        end = np.zeros(20, dtype=np.float32)
        end[self.stand.ids] = self.stand.default
        n = int(round(2.0 / dt))
        print("Ramping to the 0816 default stance over 2.0 s ...", flush=True)
        for i in range(n):
            # The e-stop and the state-validity checks must hold DURING the
            # ramp too, not only in run(): read_state re-validates age/tilt
            # and raises on a stale or falling robot.
            self._check_startup_select()
            state = self.backend.read_state()
            alpha = i / max(n - 1, 1)
            q = (1 - alpha) * start_q + alpha * end
            self.backend.write_command(PolicyCommand(q.astype(np.float32),
                                                     self.stand.kp.copy(), self.stand.kd.copy()),
                                       state_timestamp=state.extra["received_at"])
            time.sleep(dt)
        self._check_startup_select()
        stick = (abs(self.remote_controller.lx) + abs(self.remote_controller.ly)
                 + abs(self.remote_controller.rx))
        if stick > 0.1:
            print("Sticks are not centered. Center both sticks before pressing "
                  "[A] to start the walking policy.", flush=True)

    def default_pos_state(self):
        """Operator flow: [start] ramps to the stance; HERE the stance is held
        with PD until [A] hands control to the 0816 walking policy (matching
        deploy_real_h1_dwaq.py's three-stage start). The rig auto-start path
        skips the wait."""
        self._check_startup_select()
        if self.sim_auto_start and not self._startup_manual_restart:
            return
        dt = self.config.control_dt
        hold = np.zeros(20, dtype=np.float32)
        hold[self.stand.ids] = self.stand.default
        print("Holding the default stance (PD). Press [A] to start the 0816 "
              "walking policy — center both sticks first. [select] returns to "
              "zero torque.", flush=True)
        prev_a = self.remote_controller.button[KeyMap.A]
        while True:
            self._check_startup_select()
            state = self.backend.read_state()   # validates age / frozen tick / tilt
            self.backend.write_command(PolicyCommand(hold, self.stand.kp.copy(),
                                                     self.stand.kd.copy()),
                                       state_timestamp=state.extra["received_at"])
            a = self.remote_controller.button[KeyMap.A]
            if a == 1 and prev_a == 0:
                stick = (abs(self.remote_controller.lx) + abs(self.remote_controller.ly)
                         + abs(self.remote_controller.rx))
                if stick > 0.1:
                    print("WARNING: sticks not centered — center them, release "
                          "[A] and press again.", flush=True)
                else:
                    break
            prev_a = a
            time.sleep(dt)
        print("0816 walking policy active. Left stick = walk. Park at the "
              "calibrated 0.41 m gap, release the stick, then HOLD [R1] and "
              "press [A] to start the crossing. [B] cancels a pending switch; "
              "[select] e-stops any phase.", flush=True)

    def _save_rollout(self):
        super()._save_rollout()
        if self.rollout_path and self.rollout and self.rollout["time"]:
            arrays = dict(np.load(self.rollout_path))
            arrays.update(switch_weight=np.asarray(self.trace_weights),
                          target_kp=np.asarray(self.trace_kp),
                          target_kd=np.asarray(self.trace_kd),
                          stand_obs=np.asarray(self.trace_stand_obs),
                          state_age=np.asarray(self.trace_age),
                          target_gap=np.asarray(self.trace_gap),
                          stand_cmd=np.asarray(self.trace_cmd),
                          left_assist_weight=np.asarray(self.trace_assist_weight),
                          left_assist_knee_delta=np.asarray(self.trace_assist_knee_delta),
                          left_assist_base_target=np.asarray(self.trace_assist_base_target),
                          left_assist_target=np.asarray(self.trace_assist_target),
                          crossing_start_row=self.crossing_start_row)
            np.savez_compressed(self.rollout_path, **arrays)
            Path(self.rollout_path).with_name("switch_events.json").write_text(json.dumps(self.events, indent=2))
            meta = {"upstream": "https://github.com/JamesW546/h1switch", "commit": UPSTREAM_COMMIT,
                    "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "stand_config": str(OPTIONS.stand_config),
                    "stand_policy": str(self.stand.path),
                    "stand_policy_sha256": hashlib.sha256(self.stand.path.read_bytes()).hexdigest(),
                    "cross_policy": str(self.config.policy_path),
                    "cross_motion": str(self.config.motion_file),
                    "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(OPTIONS).items()},
                    "rejected_lowstates": self.rejected_states,
                    "single_lowcmd_publisher": True,
                    "left_knee_assist": self.cross.assist_settings.mode,
                    "left_leg_assist": asdict(self.cross.assist_settings),
                    "left_leg_assist_source": str(Path(left_assist_module.__file__).resolve()),
                    "left_leg_assist_source_sha256": hashlib.sha256(Path(left_assist_module.__file__).read_bytes()).hexdigest(),
                    "limiter_limits": "phase-based: 0816 contract limits while stand solely active, 0927 crossing limits during any blend or the crossing",
                    "entry_trigger": "remote [R1]+[A] combo (locomotion start: [A] after stance hold)" if not self.sim_prepare else "auto after --stand-seconds",
                    "history_contract": ("Native raw previous action; idle motion history reset. "
                                         "Motion clock owned: frame 0 held through the entry blend.")}
            Path(self.rollout_path).with_name("switch_config.json").write_text(json.dumps(meta, indent=2))
            directory = Path(self.rollout_path).parent
            assist_meta = {"scope": "Deployment-integrated position overlay; native raw policy history and PD gains unchanged",
                           "settings": asdict(self.cross.assist_settings),
                           "base_controller": str(Path(__file__).resolve()),
                           "base_controller_sha256": meta["controller_sha256"],
                           "assist_source": meta["left_leg_assist_source"],
                           "assist_source_sha256": meta["left_leg_assist_source_sha256"],
                           "integration": "MotionAdapter.step; one application before policy switching and final limiter",
                           "left_policy_indices": {"hip_pitch": 2, "knee": 3, "ankle": 4},
                           "kps_kds_modified": False, "samples": len(self.trace_assist_weight)}
            (directory / "left_leg_assist.json").write_text(json.dumps(assist_meta, indent=2) + "\n")
            np.savez_compressed(directory / "left_leg_assist.npz",
                                row=np.asarray(self.rollout["step"]),
                                motion_index=arrays["motion_index"],
                                weight=np.asarray(self.trace_assist_weight),
                                knee_delta=np.asarray(self.trace_assist_knee_delta),
                                base_target=np.asarray(self.trace_assist_base_target),
                                assisted_target=np.asarray(self.trace_assist_target))

    def run(self):
        dt = self.config.control_dt
        next_tick = time.monotonic()
        requested = False
        exit_requested = False
        exit_retry_at = 0.0
        handback_done = False
        rc = self.remote_controller
        # The A press that starts locomotion must not become a crossing press
        # when control passes from the stance hold to this loop.
        prev_a, prev_b = rc.button[KeyMap.A], rc.button[KeyMap.B]
        final_stand_rows = 0
        post_rows = max(0, (self.run_steps or self.motion_len + 250) - self.motion_len)
        auto_entry = self.sim_prepare          # rig loopback: auto after stand_seconds
        initial_rows = round(OPTIONS.stand_seconds / dt) if auto_entry else 0
        deadline = None                        # bounded only after the switch request
        try:
            self._check_run_select()
            state = self.backend.read_state()
            self.switcher.reset(state)
            while True:
                self._check_run_select()
                # Buttons act on RISING EDGES so that holding [A]+[B] cannot
                # cancel and immediately re-request in the same tick.
                a_now, b_now = rc.button[KeyMap.A], rc.button[KeyMap.B]
                a_press = a_now == 1 and prev_a == 0
                b_press = b_now == 1 and prev_b == 0
                prev_a, prev_b = a_now, b_now
                # [B] cancels a PENDING switch request before any blending
                # starts: the active policy keeps full control and [A] re-arms.
                if b_press and self.switcher.pending is not None:
                    cancelled = self.switcher.pending
                    self.switcher.pending = None
                    self.switcher.ready_time = 0.0
                    if cancelled == "stand":
                        exit_requested = False
                        exit_retry_at = time.monotonic() + 2.0
                    else:
                        requested = False
                    deadline = None      # a cancelled request must not time out later
                    self.events.append({"event": "cancel:pending", "row": self.counter,
                                        "target": cancelled})
                    print(f"Pending '{cancelled}' switch cancelled by [B]; "
                          f"'{self.switcher.active}' keeps full control.", flush=True)
                # B has priority even for simultaneous new R1+A+B presses;
                # holding B also suppresses any new entry request.
                if not requested and not handback_done and b_now == 0:
                    if auto_entry:
                        if self.counter >= initial_rows:
                            self.switcher.request_switch("cross")
                            requested = True
                            self.events.append({"event": "request:cross", "row": self.counter})
                    elif a_press:
                        if rc.button[KeyMap.R1] == 1:
                            self.switcher.request_switch("cross")
                            requested = True
                            self.events.append({"event": "request:cross", "row": self.counter,
                                                "trigger": "remote R1+A"})
                            print("Crossing requested by operator [R1]+[A].", flush=True)
                        else:
                            print("Hint: HOLD [R1] and press [A] to request the crossing.",
                                  flush=True)
                if requested and not handback_done and deadline is None:
                    deadline = (time.monotonic() + OPTIONS.entry_timeout
                                + self.motion_len * dt
                                + OPTIONS.entry_seconds + OPTIONS.exit_seconds + 5.0
                                + (post_rows * dt if self.run_steps is not None else 0.0))
                if deadline is not None and time.monotonic() > deadline:
                    raise RuntimeError("Handoff did not complete before the bounded trial deadline")
                if (self.cross.frame >= self.cross.motion_len and not exit_requested
                        and self.switcher.transition_target is None
                        and self.switcher.pending is None
                        and time.monotonic() >= exit_retry_at):
                    self.switcher.transition_seconds = OPTIONS.exit_seconds
                    self.switcher.max_target_gap = OPTIONS.exit_max_target_gap
                    self.switcher.stable_seconds = OPTIONS.exit_stable_seconds
                    self.switcher.request_switch("stand")
                    exit_requested = True
                    self.events.append({"event": "request:stand", "row": self.counter})

                state = self.backend.read_state()
                # The stick drives the stand policy only while it is solely
                # active (no pending request, no transition, no crossing).
                stick_active = (self.switcher.active == "stand"
                                and self.switcher.transition_target is None
                                and self.switcher.pending is None)
                if stick_active:
                    self.stand.cmd[:] = [rc.ly, -rc.lx, -rc.rx]
                else:
                    self.stand.cmd[:] = 0
                # Phase-appropriate software limits (see __init__): the 0816
                # contract limits while only it is active, the 0927 crossing
                # limits during any blend or the crossing itself.
                crossing_phase = (self.switcher.active == "cross"
                                  or self.switcher.transition_target is not None)
                lim = self.lim_cross if crossing_phase else self.lim_walk
                self.switcher.limiter.q_lower[:] = lim[0]
                self.switcher.limiter.q_upper[:] = lim[1]
                mixed, status = self.switcher.step(state)
                self._check_run_select()
                weight = status["weights"]["cross"]
                if weight > 0 and self.crossing_start_row < 0:
                    self.crossing_start_row = self.counter
                self.qj[:] = state.q[self.config.joint2motor_idx]
                self.dqj[:] = state.dq[self.config.joint2motor_idx]
                target = mixed.q[self.config.joint2motor_idx]
                self._append_rollout_sample(self.cross.sample_frame, self.cross.obs,
                                            self.cross.raw_action, self.cross.clipped_action,
                                            self.cross.raw_target, target,
                                            state.extra["quat"], state.extra["omega"])
                self.trace_weights.append(weight)
                self.trace_stand_obs.append(self.stand.obs.copy())
                self.trace_kp.append(mixed.kp[self.config.joint2motor_idx].copy())
                self.trace_kd.append(mixed.kd[self.config.joint2motor_idx].copy())
                self.trace_age.append(state.extra["age"])
                self.trace_gap.append(float(np.max(abs(self.stand.last_command.q - self.cross.last_command.q))))
                self.trace_cmd.append(self.stand.cmd.copy())
                self.trace_assist_weight.append(self.cross.assist_weight)
                self.trace_assist_knee_delta.append(self.cross.assist_knee_delta)
                self.trace_assist_base_target.append(self.cross.assist_base_target.copy())
                self.trace_assist_target.append(self.cross.assist_target.copy())
                self.backend.write_command(mixed, state_timestamp=state.extra["received_at"])
                if status["event"]:
                    self.events.append({"event": status["event"], "row": self.counter})
                    print(status["event"], flush=True)
                # Hand-back complete: disarm the deadline permanently (the
                # handback_done flag stops it from re-arming) — operator time
                # from here on (walk away, press [select]) is unlimited.
                if (exit_requested and not handback_done and deadline is not None
                        and status["active"] == "stand"
                        and status["transition_target"] is None and self.switcher.pending is None):
                    deadline = None
                    handback_done = True
                    print("Hand-back complete; deadline disarmed. Walk away, "
                          "press [select] to finish.", flush=True)
                if self.counter % 50 == 0:
                    gap = np.max(abs(self.stand.last_command.q - self.cross.last_command.q))
                    remain = (f" deadline_in={deadline - time.monotonic():.0f}s"
                              if deadline is not None else "")
                    print(f"t={self.counter * dt:.2f}s active={status['active']} pending={status['pending']} "
                          f"cross_weight={weight:.3f} frame={self.cross.sample_frame} gap={gap:.3f} "
                          f"cmd={np.round(self.stand.cmd, 2).tolist()} tilt={tilt(state.extra['quat']):.1f}"
                          f"{remain}", flush=True)
                # Rig runs terminate on their own after the post-motion stand rows.
                if (self.run_steps is not None and exit_requested
                        and status["active"] == "stand" and status["transition_target"] is None):
                    final_stand_rows += 1
                    if final_stand_rows >= post_rows:
                        break
                self.counter += 1
                next_tick += dt
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -dt:
                    next_tick = time.monotonic()
        except OperatorStop:
            self.events.append({"event": "operator:select", "row": self.counter,
                                "active": self.switcher.active})
        except KeyboardInterrupt:
            pass
        finally:
            self.backend.damping()
            self._save_rollout()


def main():
    global OPTIONS
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("net", type=str, help="network interface (lo for the loopback rig)")
    parser.add_argument("config", type=str, nargs="?", default=DEFAULT_CROSS_CONFIG,
                        help="crossing config name under the 0927 bundle deploy_real/configs")
    parser.add_argument("--cross-config", type=str, default=None,
                        help="override the crossing config path (absolute or 0927 bundle name)")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--sim-auto-start", action="store_true")
    parser.add_argument("--sim-prepare", action="store_true",
                        help="rig loopback: auto-request the crossing after --stand-seconds")
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--record-rollout", action="store_true")
    parser.add_argument("--rollout-name", type=str, default=None)
    parser.add_argument("--rollout-dir", type=str, default=None)
    parser.add_argument("--debug-state", action="store_true")
    parser.add_argument("--stand-config", type=Path,
                        default=H1_ASSET_ROOT / Path("0823/deploy/deploy_real/configs/"
                                     "h1_dwaq_0816_box3kg.yaml"))
    parser.add_argument("--stand-seconds", type=float, default=3.0)
    parser.add_argument("--entry-seconds", type=float, default=0.4)
    parser.add_argument("--exit-seconds", type=float, default=0.4)
    parser.add_argument("--entry-timeout", type=float, default=10.0,
                        help="seconds the entry gates may wait after the request")
    parser.add_argument("--stable-seconds", type=float, default=0.1)
    parser.add_argument("--max-target-gap", type=float, default=1.0)
    parser.add_argument("--exit-max-target-gap", type=float, default=2.0)
    parser.add_argument("--exit-stable-seconds", type=float, default=0.02)
    parser.add_argument("--motion-clock", choices=("owned", "blended"), default="owned")
    parser.add_argument("--entry-max-tilt", type=float, default=15.0)
    parser.add_argument("--entry-max-speed", type=float, default=2.0)
    add_assist_arguments(parser)
    args, remaining = parser.parse_known_args()
    if remaining:
        raise SystemExit(f"Unrecognized arguments: {remaining}")
    numeric = [v for v in vars(args).values() if isinstance(v, (float, int)) and not isinstance(v, bool)]
    if not np.isfinite(numeric).all() or min(numeric) < 0:
        raise SystemExit("Invalid switch configuration")
    try:
        validate_assist_settings(settings_from_options(args), args.motion_clock)
    except ValueError as error:
        raise SystemExit(str(error))
    OPTIONS = args

    if args.cross_config:
        p = Path(args.cross_config)
        config_path = p if p.is_absolute() else Path(base.__file__).resolve().parent / "configs" / p
    else:
        config_path = Path(base.__file__).resolve().parent / "configs" / args.config
    if not config_path.is_file():
        raise SystemExit(f"Crossing config not found: {config_path}")
    if args.sim_prepare and args.net != "lo":
        raise SystemExit("--sim-prepare is a loopback-rig-only flag (it auto-triggers the "
                         f"crossing); refusing to run on interface {args.net!r}")
    config = base.Motion104Config(str(config_path))
    validate_assist_contract(settings_from_options(args), config, np.load(config.motion_file))

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    torch.set_num_threads(1)
    ChannelFactoryInitialize(args.domain_id, args.net)
    rollout_path = None
    if args.record_rollout:
        name = args.rollout_name or Path(config_path).stem
        out_dir = Path(args.rollout_dir or (Path(__file__).resolve().parents[2] / "logs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = out_dir / f"{name}.npz"
        print(f"[INFO] Recording rollout to {rollout_path}", flush=True)
    print(f"[INFO] Left-leg assist: {asdict(settings_from_options(args))}", flush=True)
    controller = H1SwitchController(config,
                                    sim_auto_start=args.sim_auto_start,
                                    debug_state=args.debug_state,
                                    record_rollout=args.record_rollout,
                                    rollout_path=str(rollout_path) if rollout_path else None,
                                    run_steps=args.run_steps)
    try:
        controller.startup()
        controller.run()
    finally:
        # Startup phases (the zero-torque wait and the 2 s ramp) must also
        # leave the robot damped on Ctrl+C or an exception, not just run().
        controller.backend.damping()
    print("Exit", flush=True)


if __name__ == "__main__":
    main()
