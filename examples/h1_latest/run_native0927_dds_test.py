#!/usr/bin/env python3
"""One DDS comparison with the unchanged native 0927 H1 model and 30 cm board.

The initial default stance is PRESET; real Start/ramp/A startup is excluded.
An optional initial stiff-PD settle is an explicit simulation preparation.
The existing controller runs in a separate process and receives real SDK2
messages on loopback. No policy, controller, scene, or robot source is edited.

Example (run only when authorized):
  python -B -u run_native0927_dds_test.py run \
    --domain-id 198 --output validation/native0927_dds_comparison

Importing this module or calling load_native() does not initialize DDS.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1] / "assets/local/h1/0927"
SCENE = BUNDLE / "unitree_mujoco/unitree_robots/h1/scene_coretest_stairhome_obstacle0p30_yawm76p5.xml"
ROBOT = SCENE.parent / "h1_coretest_stairhome_0927.xml"
CROSS_CONFIG = BUNDLE / "deploy/deploy_real/configs/h1_motion_104_coretestvideo_gmr_stairhome_obstacle0p30_yawm76p5_model30000.yaml"
STAND_CONFIG = HERE.parents[1] / "assets/local/h1/0823/deploy/deploy_real/configs/h1_dwaq_0816_box3kg.yaml"
CONTROLLER = HERE / "deploy_real_h1_motion_104_h1switch_0927.py"
BOARD_NAME = "coretest_stairhome_obstacle_0p30"
JOINTS = (
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
    "left_ankle_joint", "right_hip_yaw_joint", "right_hip_roll_joint",
    "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint", "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
)
MOTOR_INDICES = (7, 3, 4, 5, 10, 8, 0, 1, 2, 11, 6, 16, 17, 18, 19, 12, 13, 14, 15)
SCOPE = "Preset native motion-frame-0/default stance; real Start, pose ramp and A startup excluded"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sensor(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise ValueError(f"Missing native sensor {name}")
    start = int(model.sensor_adr[sid])
    return data.sensordata[start:start + int(model.sensor_dim[sid])].copy()


def prepare_effective_scene(output, board_normal_offset=0.):
    """Compile a frozen XML snapshot with only the board position varied.

    Expand the original include and use an absolute asset directory so the
    snapshot can be loaded from the result directory. Recompilation rebuilds
    static collision acceleration structures for the actual board position.
    The original scene and robot XML files are never edited.
    """
    if not np.isfinite(board_normal_offset):
        raise ValueError("Board normal offset must be finite")
    scene = ET.parse(SCENE).getroot()
    includes = list(scene.findall("include"))
    if len(includes) != 1 or (SCENE.parent / includes[0].get("file", "")).resolve() != ROBOT.resolve():
        raise ValueError("Unexpected native scene include")
    index = list(scene).index(includes[0])
    scene.remove(includes[0])
    robot = ET.parse(ROBOT).getroot()
    for child in robot:
        scene.insert(index, copy.deepcopy(child))
        index += 1
    for compiler in scene.findall("compiler"):
        for attribute in ("meshdir", "texturedir"):
            value = compiler.get(attribute)
            if value:
                compiler.set(attribute, str((ROBOT.parent / value).resolve()))
    boards = scene.findall(".//geom[@name='" + BOARD_NAME + "']")
    if len(boards) != 1:
        raise ValueError("Expected one native board")
    board = boards[0]
    position = np.fromstring(board.get("pos", ""), sep=" ")
    quat = np.fromstring(board.get("quat", ""), sep=" ")
    if position.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[position, quat]).all():
        raise ValueError("Invalid source board pose")
    w, x, y, z = quat / np.linalg.norm(quat)
    normal = np.asarray([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
    position += float(board_normal_offset) * normal
    board.set("pos", " ".join(format(float(value), ".17g") for value in position))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "effective_scene.xml"
    content = ET.tostring(scene, encoding="unicode") + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError("Existing effective scene differs from the requested board offset")
    else:
        path.write_text(content)
    return path.resolve()


def load_native(initial_forward_offset=0., disable_board_contact=False, scene_path=SCENE):
    """Load and initialize the native model, without SDK imports or DDS."""
    cfg = yaml.safe_load(CROSS_CONFIG.read_text())
    for key in ("policy_path", "motion_file"):
        cfg[key] = cfg[key].replace("{LEGGED_GYM_ROOT_DIR}", str(BUNDLE))
    with np.load(cfg["motion_file"]) as recording:
        motion = {key: value.copy() for key, value in recording.items()}
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    jids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINTS])
    if (jids < 0).any():
        raise ValueError("Native model is missing a controlled joint")
    qidx = np.asarray(model.jnt_qposadr[jids], dtype=int)
    vidx = np.asarray(model.jnt_dofadr[jids], dtype=int)
    aidx = np.asarray([next((a for a in range(model.nu)
                           if int(model.actuator_trnid[a, 0]) == int(jid)), -1)
                       for jid in jids], dtype=int)
    if ((aidx < 0).any() or len(set(aidx.tolist())) != 19
            or tuple(cfg["joint2motor_idx"]) != MOTOR_INDICES
            or model.nu != 20 or model.nq != 27 or model.nv != 26
            or abs(model.opt.timestep - .002) > 1e-9
            or abs(float(cfg["control_dt"]) - .02) > 1e-9):
        raise ValueError("Unexpected native joint, actuator, DDS, or clock contract")
    unused = np.asarray([a for a in range(model.nu) if a not in set(aidx.tolist())], dtype=int)
    dummy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "not_use_joint")
    if len(unused) != 1 or int(model.actuator_trnid[unused[0], 0]) != dummy:
        raise ValueError("The only unused native actuator must be not_use_joint")
    if (np.any(np.asarray(cfg["joint_limit_lower"]) < model.jnt_range[jids, 0] - 1e-6)
            or np.any(np.asarray(cfg["joint_limit_upper"]) > model.jnt_range[jids, 1] + 1e-6)):
        raise ValueError("Controller limits exceed native model limits")
    # Preserve the dummy joint and every model parameter; set only the robot state.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:3] = motion["body_pos_w"][0, 0]
    data.qpos[3:7] = motion["body_quat_w"][0, 0]
    data.qpos[qidx] = motion["joint_pos"][0]
    data.qvel[:] = 0
    data.ctrl[:] = 0
    if not np.allclose(data.qpos[qidx], cfg["default_angles"], atol=1e-6):
        raise ValueError("Native first motion frame differs from the nominal default stance")
    stand_cfg = yaml.safe_load(STAND_CONFIG.read_text())
    native_slots = dict(zip(MOTOR_INDICES, data.qpos[qidx]))
    if not np.allclose([native_slots[i] for i in stand_cfg["joint2motor_idx"]],
                       stand_cfg["default_angles"], atol=1e-6):
        raise ValueError("Native first frame differs from the supplied 0816 default stance")
    mujoco.mj_forward(model, data)
    board = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, BOARD_NAME)
    if board < 0 or not np.allclose(model.geom_size[board], [2.5, .02, .15]):
        raise ValueError("Unexpected native 30 cm board geometry")
    # Diagnostic interventions are explicit and never change source XML.
    # Translation follows the fixed board normal toward its far side.
    data.qpos[:3] += float(initial_forward_offset) * data.geom_xmat[board].reshape(3, 3)[:, 1]
    mujoco.mj_forward(model, data)
    if any(board in (c.geom1, c.geom2) for c in data.contact):
        raise ValueError("Preset initial pose intersects the native board")
    if disable_board_contact:
        model.geom_contype[board] = 0
        model.geom_conaffinity[board] = 0
        mujoco.mj_forward(model, data)
    return cfg, motion, model, data, qidx, vidx, aidx, unused, board


class NativeSimulator:
    def __init__(self, args):
        self.scene_path = prepare_effective_scene(args.output, args.board_normal_offset)
        (self.cfg, self.motion, self.model, self.data, self.qidx, self.vidx,
         self.aidx, self.unused_aidx, self.board_id) = load_native(args.initial_forward_offset, args.disable_board_contact, self.scene_path)
        self.args = args
        self.initial_q = self.data.qpos[self.qidx].copy()
        self.command = None
        self.command_count = self.bad_crc_count = self.bad_mode_count = 0
        self.last_command_wall = 0.
        self.alive = True
        self.rows = {key: [] for key in ("time", "qpos", "qvel", "ctrl", "command_count",
                    "command_age", "board_contacts", "left_ankle_pos", "right_ankle_pos",
                    "left_foot_pos", "right_foot_pos")}
        self.ankles = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_link")
                       for s in ("left", "right")]
        self.feet = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_foot")
                     for s in ("left", "right")]
        self._settle()
        # Real SDK2 messages and the original GO mode/CRC contract. Initialization
        # occurs only in the simulator child process, exclusively on loopback.
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_, unitree_go_msg_dds__SportModeState_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
        from unitree_sdk2py.utils.crc import CRC
        self.crc = CRC()
        self.low_msg = unitree_go_msg_dds__LowState_()
        self.high_msg = unitree_go_msg_dds__SportModeState_()
        # The unused wire motor slot 9 has no controlled joint and remains zero.
        ChannelFactoryInitialize(args.domain_id, "lo")
        self.low_pub = ChannelPublisher(self.cfg["lowstate_topic"], LowState_)
        self.low_pub.Init()
        self.high_pub = ChannelPublisher(self.cfg["highstate_topic"], SportModeState_)
        self.high_pub.Init()
        self.cmd_sub = ChannelSubscriber(self.cfg["lowcmd_topic"], LowCmd_)
        self.cmd_sub.Init(self.on_command, 10)

    def _settle(self):
        count = int(round(self.args.settle_seconds / self.model.opt.timestep))
        for step in range(count):
            mujoco.mj_forward(self.model, self.data)
            tau = 400. * (self.initial_q - self.data.qpos[self.qidx]) - 10. * self.data.qvel[self.vidx]
            self.data.ctrl[:] = 0
            self.data.ctrl[self.aidx] = np.clip(tau, self.model.actuator_ctrlrange[self.aidx, 0],
                                               self.model.actuator_ctrlrange[self.aidx, 1])
            if step % 10 == 0:
                self.record(float("inf"))
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:] = 0

    def on_command(self, message):
        if message.crc != self.crc.Crc(message):
            self.bad_crc_count += 1
            return
        if (list(message.head) != [0xFE, 0xEF] or message.level_flag != 0xFF
                or any(message.motor_cmd[i].mode != (1 if i in self.cfg["weak_motor"] else 0x0A)
                       for i in MOTOR_INDICES)):
            self.bad_mode_count += 1
            return
        self.command = message
        self.command_count += 1
        self.last_command_wall = time.monotonic()

    def publish_state(self):
        d, m = self.data, self.model
        for j, wire_slot in enumerate(MOTOR_INDICES):
            motor = self.low_msg.motor_state[wire_slot]
            motor.q = float(d.qpos[self.qidx[j]])
            motor.dq = float(d.qvel[self.vidx[j]])
            motor.ddq = float(d.qacc[self.vidx[j]])
            motor.tau_est = float(d.actuator_force[self.aidx[j]])
        imu = self.low_msg.imu_state
        imu.quaternion[:] = [float(v) for v in sensor(m, d, "imu_quat")]
        imu.gyroscope[:] = [float(v) for v in sensor(m, d, "imu_gyro")]
        imu.accelerometer[:] = [float(v) for v in sensor(m, d, "imu_acc")]
        self.low_msg.tick = max(1, int(round(d.time * 1000)))
        self.low_msg.crc = self.crc.Crc(self.low_msg)
        self.low_pub.Write(self.low_msg)
        self.high_msg.position[:] = [float(v) for v in d.qpos[:3]]
        self.high_msg.velocity[:] = [float(v) for v in d.qvel[:3]]
        self.high_msg.imu_state.quaternion[:] = imu.quaternion
        self.high_msg.imu_state.gyroscope[:] = imu.gyroscope
        self.high_pub.Write(self.high_msg)

    def apply_command(self):
        d, m, cmd = self.data, self.model, self.command
        age = time.monotonic() - self.last_command_wall if cmd is not None else float("inf")
        if cmd is None:
            tau = 120. * (self.initial_q - d.qpos[self.qidx]) - 4. * d.qvel[self.vidx]
        elif age > .5:
            tau = -8. * d.qvel[self.vidx]
        else:
            tau = np.asarray([cmd.motor_cmd[s].tau
                              + cmd.motor_cmd[s].kp * (cmd.motor_cmd[s].q - d.qpos[self.qidx[j]])
                              + cmd.motor_cmd[s].kd * (cmd.motor_cmd[s].dq - d.qvel[self.vidx[j]])
                              for j, s in enumerate(MOTOR_INDICES)])
        d.ctrl[:] = 0  # includes the native not_use_joint actuator
        d.ctrl[self.aidx] = np.clip(tau, m.actuator_ctrlrange[self.aidx, 0], m.actuator_ctrlrange[self.aidx, 1])
        return age

    def record(self, age):
        d = self.data
        values = (d.time, d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), self.command_count, age,
                  sum(self.board_id in (c.geom1, c.geom2) for c in d.contact),
                  d.xpos[self.ankles[0]].copy(), d.xpos[self.ankles[1]].copy(),
                  d.site_xpos[self.feet[0]].copy(), d.site_xpos[self.feet[1]].copy())
        for key, value in zip(self.rows, values):
            self.rows[key].append(value)

    def save(self):
        self.args.output.mkdir(parents=True, exist_ok=True)
        m, d = self.model, self.data
        board_rot = d.geom_xmat[self.board_id].reshape(3, 3)
        np.savez_compressed(self.args.output / "sim.npz", **{k: np.asarray(v) for k, v in self.rows.items()},
                            native_scene=1, scene=str(self.scene_path), scene_sha256=sha256(self.scene_path),
                            source_scene=str(SCENE), source_scene_sha256=sha256(SCENE),
                            robot_xml=str(ROBOT), robot_xml_sha256=sha256(ROBOT),
                            board_name=BOARD_NAME, board_normal_axis=1,
                            board_center=d.geom_xpos[self.board_id].copy(), board_normal=board_rot[:, 1].copy(),
                            board_x=float(d.geom_xpos[self.board_id, 0]),
                            board_y=float(d.geom_xpos[self.board_id, 1]),
                            board_yaw_deg=float(np.degrees(np.arctan2(board_rot[1, 0], board_rot[0, 0]))),
                            board_thickness=.04, board_height=.30, board_half_size=m.geom_size[self.board_id].copy(),
                            controlled_joint_names=np.asarray(JOINTS), qpos_indices=self.qidx, dof_indices=self.vidx,
                            actuator_indices=self.aidx, motor_indices=np.asarray(MOTOR_INDICES),
                            unused_actuator_indices=self.unused_aidx, unused_wire_slot=9,
                            command_total=self.command_count, bad_crc_total=self.bad_crc_count,
                            bad_mode_total=self.bad_mode_count, domain_id=self.args.domain_id,
                            crossing_config=str(CROSS_CONFIG), crossing_config_sha256=sha256(CROSS_CONFIG),
                            motion_file=self.cfg["motion_file"], motion_sha256=sha256(self.cfg["motion_file"]),
                            policy_path=self.cfg["policy_path"], policy_sha256=sha256(self.cfg["policy_path"]),
                            controller_sha256=sha256(CONTROLLER), simulator_sha256=sha256(__file__),
                            left_leg_assist_source_sha256=sha256(CONTROLLER.parent / "left_leg_assist.py"),
                            controller_entrypoint=str(self.args.controller_script.resolve()),
                            controller_entrypoint_sha256=sha256(self.args.controller_script),
                            robot_mass_kg=float(m.body_subtreemass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]),
                            motor_gain=1., motor_time_constant=0., inertia_urdf="", urdf_joint_limits_applied=0,
                            initial_pose="native_default_preset", initialization_scope=SCOPE,
                            settle_seconds=self.args.settle_seconds, physics_dt=m.opt.timestep,
                            initial_forward_offset_m=self.args.initial_forward_offset,
                            board_normal_offset_m=self.args.board_normal_offset,
                            board_contact_disabled=int(self.args.disable_board_contact),
                            control_dt=float(self.cfg["control_dt"]), mujoco_version=mujoco.__version__)

    def run(self):
        began = next_tick = time.monotonic()
        dt, step = self.model.opt.timestep, 0
        print(f"Native 0927 DDS comparison: domain={self.args.domain_id}, dt={dt}; {SCOPE}; "
              f"initial stiff settle={self.args.settle_seconds}s", flush=True)
        try:
            while self.alive and time.monotonic() - began < self.args.max_seconds:
                mujoco.mj_forward(self.model, self.data)
                if step % 5 == 0:
                    self.publish_state()
                age = self.apply_command()
                if step % 10 == 0:
                    self.record(age)
                # Freeze physical time during controller loading, as in the original rig.
                if self.command is not None:
                    mujoco.mj_step(self.model, self.data)
                if not np.isfinite(self.data.qpos).all():
                    raise RuntimeError("Native MuJoCo pose became nonfinite")
                if step % 250 == 0:
                    print(f"t={self.data.time:.2f}s z={self.data.qpos[2]:.3f} cmds={self.command_count} "
                          f"bad_crc={self.bad_crc_count} bad_mode={self.bad_mode_count}", flush=True)
                step += 1
                next_tick += dt
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        finally:
            self.save()
            print(f"Saved {len(self.rows['time'])} samples to {self.args.output / 'sim.npz'}", flush=True)


def stop_child(process, first_signal, timeout):
    if process is None or process.poll() is not None:
        return
    process.send_signal(first_signal)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def launch(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "sim.npz").exists() or (args.output / "controller.npz").exists():
        raise FileExistsError("Use a fresh output directory for the single native comparison")
    scene_path = prepare_effective_scene(args.output, args.board_normal_offset)
    cfg, motion, model, data, qidx, vidx, aidx, unused, board = load_native(args.initial_forward_offset, args.disable_board_contact, scene_path)
    stand_cfg = yaml.safe_load(STAND_CONFIG.read_text())
    stand_root = STAND_CONFIG.parents[3]
    stand_policy = Path(stand_cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(stand_root)))
    controller_script = args.controller_script.resolve()
    if not controller_script.is_file():
        raise FileNotFoundError(controller_script)
    manifest = {"scope": SCOPE, "network": "lo", "domain_id": args.domain_id,
                "scene": str(scene_path), "scene_sha256": sha256(scene_path),
                "source_scene": str(SCENE), "source_scene_sha256": sha256(SCENE),
                "robot_xml": str(ROBOT), "robot_xml_sha256": sha256(ROBOT),
                "controller": str(controller_script), "controller_sha256": sha256(controller_script),
                "base_controller": str(CONTROLLER), "base_controller_sha256": sha256(CONTROLLER),
                "left_leg_assist_source": str(CONTROLLER.parent / "left_leg_assist.py"),
                "left_leg_assist_source_sha256": sha256(CONTROLLER.parent / "left_leg_assist.py"),
                "controller_extra": args.controller_extra,
                "simulator": str(Path(__file__).resolve()), "simulator_sha256": sha256(__file__),
                "config_sha256": sha256(CROSS_CONFIG), "stand_config_sha256": sha256(STAND_CONFIG),
                "cross_policy_sha256": sha256(cfg["policy_path"]), "motion_sha256": sha256(cfg["motion_file"]),
                "stand_policy_sha256": sha256(stand_policy), "mujoco": mujoco.__version__,
                "numpy": np.__version__, "python": sys.version, "executable": sys.executable,
                "physics_dt": float(model.opt.timestep), "control_dt": float(cfg["control_dt"]),
                "nq": model.nq, "nv": model.nv, "nu": model.nu,
                "qpos_indices": qidx.tolist(), "dof_indices": vidx.tolist(), "actuator_indices": aidx.tolist(),
                "unused_actuator_indices": unused.tolist(), "unused_wire_motor_slot": 9,
                "board_center": data.geom_xpos[board].tolist(), "board_half_size": model.geom_size[board].tolist(),
                "board_normal_axis": 1, "settle_seconds": args.settle_seconds,
                "source_parameters_modified": False,
                "initial_forward_offset_m": args.initial_forward_offset,
                "board_normal_offset_m": args.board_normal_offset,
                "effective_scene_parameters_modified": bool(args.board_normal_offset),
                "board_contact_disabled": args.disable_board_contact,
                "compiled_collision_parameters_modified": args.disable_board_contact,
                "diagnostic_scope": "Board contact disabled: not an obstacle crossing test" if args.disable_board_contact
                                    else "Original board contact; translated initial robot state" if args.initial_forward_offset
                                    else "Board translated along its normal; original initial robot state" if args.board_normal_offset
                                    else "Unmodified native scene and original initial robot state"}
    (args.output / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sim_cmd = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "sim",
               "--domain-id", str(args.domain_id), "--output", str(args.output),
               "--max-seconds", str(args.max_seconds), "--settle-seconds", str(args.settle_seconds)]
    sim_cmd += ["--initial-forward-offset", str(args.initial_forward_offset)]
    sim_cmd += ["--board-normal-offset", str(args.board_normal_offset)]
    sim_cmd += ["--controller-script", str(controller_script)]
    if args.disable_board_contact:
        sim_cmd.append("--disable-board-contact")
    ctrl_cmd = [sys.executable, "-B", "-u", str(controller_script), "lo", "--domain-id", str(args.domain_id),
                "--cross-config", str(CROSS_CONFIG), "--sim-auto-start", "--sim-prepare",
                "--run-steps", str(len(motion["joint_pos"]) + 250), "--record-rollout",
                "--rollout-dir", str(args.output), "--rollout-name", "controller"]
    ctrl_cmd.extend(args.controller_extra)
    status = {"completed": False, "physical_crossing_verified": False,
              "completion_scope": "controller process exit status only; both-foot passage needs independent geometry audit",
              "initialization_scope": SCOPE, "error": None, "simulator_command": sim_cmd,
              "controller_command": ctrl_cmd, "controller_exit": None, "simulator_exit": None,
              "domain_id": args.domain_id}
    status["diagnostic_interventions"] = {"initial_forward_offset_m": args.initial_forward_offset,
                                         "board_normal_offset_m": args.board_normal_offset,
                                         "board_contact_disabled": args.disable_board_contact}
    sim = controller = None
    try:
        with (args.output / "sim.log").open("w") as sim_log, (args.output / "controller.log").open("w") as ctrl_log:
            try:
                environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                sim = subprocess.Popen(sim_cmd, stdout=sim_log, stderr=subprocess.STDOUT, env=environment, cwd=str(HERE))
                time.sleep(1.)
                if sim.poll() is not None:
                    raise RuntimeError("Native simulator exited during initialization; see sim.log")
                controller = subprocess.Popen(ctrl_cmd, stdout=ctrl_log, stderr=subprocess.STDOUT,
                                              env=environment, cwd=str(HERE))
                deadline = time.monotonic() + max(1., args.max_seconds - 1.)
                while controller.poll() is None:
                    if sim.poll() is not None:
                        raise RuntimeError("Native simulator exited before controller completion")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Native comparison reached its bounded trial timeout")
                    time.sleep(.1)
                status["controller_exit"] = controller.returncode
                if controller.returncode != 0:
                    raise RuntimeError(f"Controller exited with status {controller.returncode}; see controller.log")
                time.sleep(.2)  # record terminal damping without presenting it as stable hand-back
                status["completed"] = True
            finally:
                stop_child(controller, signal.SIGINT, 5)
                stop_child(sim, signal.SIGINT, 10)
                if controller is not None:
                    status["controller_exit"] = controller.returncode
                if sim is not None:
                    status["simulator_exit"] = sim.returncode
    except BaseException as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Native DDS comparison saved to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "sim"))
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--domain-id", type=int, default=198, help="isolated DDS domain; interface is always lo")
    parser.add_argument("--max-seconds", type=float, default=30.)
    parser.add_argument("--settle-seconds", type=float, default=.5,
                        help="preset stiff-PD preparation before DDS starts, explicitly excluded from real startup")
    parser.add_argument("--initial-forward-offset", type=float, default=0.,
                        help="diagnostic robot-state translation in metres along the fixed board normal")
    parser.add_argument("--board-normal-offset", type=float, default=0.,
                        help="board translation in metres along its normal; positive increases starting distance, robot unchanged")
    parser.add_argument("--disable-board-contact", action="store_true",
                        help="diagnostic ONLY: disable compiled board collision; cannot count as crossing success")
    parser.add_argument("--controller-script", type=Path, default=CONTROLLER,
                        help="diagnostic alternative controller entrypoint; default unchanged deployment source")
    parser.add_argument("--controller-extra", nargs=argparse.REMAINDER, default=[],
                        help="remaining arguments appended to diagnostic controller, placed last")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if (not 1 <= args.domain_id <= 232 or not np.isfinite([args.max_seconds, args.settle_seconds, args.initial_forward_offset, args.board_normal_offset]).all()
            or args.max_seconds <= 0 or args.settle_seconds < 0):
        parser.error("Use an isolated domain 1..232, positive max-seconds and nonnegative settle-seconds")
    if args.mode == "run":
        launch(args)
    else:
        sim = NativeSimulator(args)
        signal.signal(signal.SIGINT, lambda *_: setattr(sim, "alive", False))
        signal.signal(signal.SIGTERM, lambda *_: setattr(sim, "alive", False))
        sim.run()


if __name__ == "__main__":
    main()
