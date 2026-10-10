"""Load policy assets and robot models independently for separate DDS processes."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import yaml

from .simulation import Simulation

ROOT = Path(__file__).resolve().parents[1]
H1_JOINTS = ["left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint",
             "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint",
             "torso_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
             "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"]
G1_POLICIES = ROOT/"assets/public/unitree_rl_lab/deploy/robots/g1_29dof/config/policy"


def h1_data():
    bundle = ROOT/"assets/local/h1/0927"
    files = list((bundle/"deploy/deploy_real/configs").glob("*model30000.yaml"))
    if len(files) != 1:
        raise FileNotFoundError("Import the H1 assets first; see scripts/import_h1_assets.py --help")
    cfg = yaml.safe_load(files[0].read_text())
    for key in ["policy_path", "motion_file"]:
        cfg[key] = cfg[key].replace("{LEGGED_GYM_ROOT_DIR}", str(bundle))
    for key in ["default_angles", "action_scale", "joint_limit_lower", "joint_limit_upper"]:
        cfg[key] = np.asarray(cfg[key], dtype=np.float32)
    with np.load(cfg["motion_file"], allow_pickle=False) as data:
        reference = {key: value.astype(np.float32) for key, value in data.items()}
    return SimpleNamespace(**cfg), reference, bundle


def build_policies(args):
    if args.robot == "g1":
        from .g1 import G1Mimic, G1Policy
        return (G1Policy("locomotion", G1_POLICIES/"velocity/v0"),
                G1Mimic(G1_POLICIES/"mimic/dance_102", args.motion_start, args.motion_end))
    import torch
    from .h1 import Dwaq0816Adapter, MotionAdapter
    torch.set_num_threads(1)
    cfg, reference, _ = h1_data()
    owner = SimpleNamespace(config=cfg, motion=reference,
                            policy=torch.jit.load(cfg.policy_path, map_location="cpu").eval(),
                            options=SimpleNamespace(motion_clock="owned", entry_max_tilt=15., entry_max_speed=2.,
                                                    left_assist_mode=args.left_assist))
    return (Dwaq0816Adapter(ROOT/"assets/local/h1/0823/deploy/deploy_real/configs/h1_dwaq_0816_box3kg.yaml"),
            MotionAdapter(owner))


def build_simulation(args):
    if args.robot == "g1":
        cfg = yaml.safe_load((G1_POLICIES/"velocity/v0/params/deploy.yaml").read_text())
        sim = Simulation(ROOT/"assets/public/unitree_mujoco/unitree_robots/g1/scene_29dof.xml", 29)
        q = np.zeros(29)
        q[cfg["joint_ids_map"]] = cfg["default_joint_pos"]
        sim.initialize(q, [0, 0, .79])
        return sim
    cfg, reference, bundle = h1_data()
    xml = bundle/"unitree_mujoco/unitree_robots/h1/scene_coretest_stairhome_obstacle0p30_yawm76p5.xml"
    sim = Simulation(xml, 20, H1_JOINTS, cfg.joint2motor_idx)
    q = np.zeros(20)
    q[cfg.joint2motor_idx] = cfg.default_angles
    sim.initialize(q, reference["body_pos_w"][0, 0], reference["body_quat_w"][0, 0])
    # Preserve the v0.1 flat-ground task and native robot dynamics.
    board = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "coretest_stairhome_obstacle_0p30")
    if board >= 0:
        sim.model.geom_contype[board] = sim.model.geom_conaffinity[board] = 0
        sim.model.geom_rgba[board, 3] = 0
    return sim


def build_g1(args):
    options = SimpleNamespace(**dict(vars(args), robot="g1"))
    return build_simulation(options), *build_policies(options)


def build_h1(args):
    options = SimpleNamespace(**dict(vars(args), robot="h1"))
    return build_simulation(options), *build_policies(options)
