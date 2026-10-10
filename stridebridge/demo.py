"""Headless continuous locomotion -> imitation -> locomotion demonstrations."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import yaml

from .geometry import tilt
from .simulation import Simulation
from .switcher import CommandLimiter, PolicySwitcher

ROOT = Path(__file__).resolve().parents[1]
H1_JOINTS = ["left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint", "left_ankle_joint",
             "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint", "right_ankle_joint",
             "torso_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
             "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"]


def build_g1(args):
    from .g1 import G1Mimic, G1Policy
    assets = ROOT/"assets/public"
    policies = assets/"unitree_rl_lab/deploy/robots/g1_29dof/config/policy"
    walk = G1Policy("locomotion", policies/"velocity/v0")
    motion = G1Mimic(policies/"mimic/dance_102", args.motion_start, args.motion_end)
    sim = Simulation(assets/"unitree_mujoco/unitree_robots/g1/scene_29dof.xml", 29)
    q = np.zeros(29)
    q[walk.ids] = walk.default
    sim.initialize(q, [0, 0, .79])
    return sim, walk, motion


def build_h1(args):
    import torch
    from .h1 import Dwaq0816Adapter, MotionAdapter
    torch.set_num_threads(1)
    assets = ROOT/"assets/local/h1"
    bundle = assets/"0927"
    configs = bundle/"deploy/deploy_real/configs"
    files = list(configs.glob("*model30000.yaml"))
    if len(files) != 1:
        raise FileNotFoundError("Import the H1 assets first; see scripts/import_h1_assets.py --help")
    cfg = yaml.safe_load(files[0].read_text())
    for key in ["policy_path", "motion_file"]:
        cfg[key] = cfg[key].replace("{LEGGED_GYM_ROOT_DIR}", str(bundle))
    for key in ["default_angles", "action_scale", "joint_limit_lower", "joint_limit_upper"]:
        cfg[key] = np.asarray(cfg[key], dtype=np.float32)
    cfg = SimpleNamespace(**cfg)
    with np.load(cfg.motion_file, allow_pickle=False) as data:
        reference = {k: v.astype(np.float32) for k, v in data.items()}
    owner = SimpleNamespace(config=cfg, motion=reference,
                            policy=torch.jit.load(cfg.policy_path, map_location="cpu").eval(),
                            options=SimpleNamespace(motion_clock="owned", entry_max_tilt=15., entry_max_speed=2.,
                                                    left_assist_mode=args.left_assist))
    motion = MotionAdapter(owner)
    walk = Dwaq0816Adapter(assets/"0823/deploy/deploy_real/configs/h1_dwaq_0816_box3kg.yaml")
    xml = bundle/"unitree_mujoco/unitree_robots/h1/scene_coretest_stairhome_obstacle0p30_yawm76p5.xml"
    sim = Simulation(xml, 20, H1_JOINTS, cfg.joint2motor_idx)
    q = np.zeros(20)
    q[cfg.joint2motor_idx] = cfg.default_angles
    sim.initialize(q, reference["body_pos_w"][0, 0], reference["body_quat_w"][0, 0])
    # Flat-ground switching keeps the original robot dynamics; only the board is removed.
    board = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "coretest_stairhome_obstacle_0p30")
    if board >= 0:
        sim.model.geom_contype[board] = sim.model.geom_conaffinity[board] = 0
        sim.model.geom_rgba[board, 3] = 0
    return sim, walk, motion


def run(args):
    if args.output.exists():
        raise FileExistsError("Use a new output directory to preserve previous trials: " + str(args.output))
    asset_manifest = ROOT/("assets/manifest.json" if args.robot == "g1" else "assets/local/h1/manifest.json")
    assets = json.loads(asset_manifest.read_text())
    groups = [(ROOT/"assets/public"/s["name"], s["files"]) for s in assets["sources"]] if args.robot == "g1" else [(ROOT/"assets/local/h1", assets)]
    for base, files in groups:
        for item in files:
            path = base/item["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Asset checksum mismatch: " + str(path))
    sim, walk, motion = build_g1(args) if args.robot == "g1" else build_h1(args)
    dt = .02
    # H1 entry/exit settings match the latest controller. G1 is calibrated separately.
    limiter = CommandLimiter(sim.motor_count, sim.lower, sim.upper)
    switch = PolicySwitcher(walk, motion, walk.name, args.blend, dt,
                            stable_seconds=.1, max_target_gap=args.entry_gap, limiter=limiter)
    switch.reset(sim.read())
    args.output.mkdir(parents=True)
    events, rows = [], []
    entered, returned, requested_exit = False, None, False
    request_at = 6. if args.scenario == "walking" else 3.
    failure = None
    try:
        for _ in range(round(args.max_seconds/dt)):
            state = sim.read()
            t = state.timestamp
            if tilt(state.extra["quat"]) >= 45 or state.extra["height"] < (.45 if args.robot == "g1" else .65):
                failure = "fall_threshold"
                break
            command = np.zeros(3, dtype=np.float32)
            if args.scenario == "walking":
                if 1 <= t < request_at-1.5 or (returned is not None and t >= returned+1.):
                    command[0] = args.speed
            if args.robot == "g1":
                walk.command[:] = command if switch.active == walk.name and not switch.pending and not switch.transition_target else 0
            else:
                walk.cmd[:] = command / walk.max_cmd if switch.active == walk.name and not switch.pending and not switch.transition_target else 0
            if t >= request_at and not entered:
                switch.request_switch(motion.name)
                events.append({"time": t, "event": "request:motion"})
                entered = True
            done = motion.done if args.robot == "g1" else motion.frame >= motion.motion_len
            if done and not requested_exit and switch.transition_target is None:
                switch.max_target_gap = args.exit_gap
                switch.stable_seconds = .02
                switch.request_switch(walk.name)
                events.append({"time": t, "event": "request:locomotion"})
                requested_exit = True
            if args.robot == "h1":
                cfg = motion.cfg if switch.active == motion.name or switch.transition_target else SimpleNamespace(**walk.cfg)
                ids = motion.ids if cfg is motion.cfg else walk.ids
                limiter.q_lower[ids] = cfg.joint_limit_lower
                limiter.q_upper[ids] = cfg.joint_limit_upper
            mixed, status = switch.step(state)
            if status["event"]:
                events.append({"time": t, "event": status["event"]})
                print(round(t, 2), status["event"], flush=True)
                if requested_exit and status["active"] == walk.name:
                    returned = t
            rows.append({"time": t, "qpos": sim.data.qpos.copy(), "qvel": sim.data.qvel.copy(),
                         "tilt": tilt(state.extra["quat"]), "height": state.extra["height"],
                         "velocity": state.extra["velocity"], "command": command.copy(),
                         "motion_weight": status["weights"][motion.name], "target": mixed.q.copy(),
                         "motion_time": motion.time if args.robot == "g1" else min(motion.frame, motion.motion_len)*dt})
            sim.step(mixed)
            if returned is not None and t-returned >= args.post_seconds-1e-8:
                break
        else:
            failure = "transition_timeout"
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]} if rows else {}
    np.savez_compressed(args.output/"rollout.npz", **arrays)
    final = sim.read()
    final_tilt = tilt(final.extra["quat"])
    if final_tilt >= 45 or final.extra["height"] < (.45 if args.robot == "g1" else .65):
        failure = failure or "fall_threshold"
    post = [r for r in rows if returned is not None and r["time"] >= returned+1.]
    post_max = max([r["tilt"] for r in post]+[final_tilt]) if post else None
    post_duration = float(rows[-1]["time"]-returned) if post and returned is not None else 0.
    completed = returned is not None and post_duration >= args.post_seconds-dt-1e-8
    measured_vx = float(np.mean([r["velocity"][0] for r in post])) if post else None
    speed_mean = float(np.mean([np.linalg.norm(r["velocity"][:2]) for r in post])) if post else None
    vx_rmse = float(np.sqrt(np.mean([(r["velocity"][0]-r["command"][0])**2 for r in post]))) if post else None
    speed_ok = bool(post and (speed_mean < .15 if args.scenario == "standing"
                             else abs(measured_vx-args.speed) <= max(.15, .5*args.speed)))
    success = bool(failure is None and completed and post_max is not None and post_max < 15. and speed_ok)
    if not success and failure is None:
        failure = "post_transition_acceptance"
    report = {"robot": args.robot, "scenario": args.scenario, "success": success,
              "failure": failure, "switch_completed": returned is not None,
              "duration_s": float(sim.data.time), "post_transition_duration_s": post_duration,
              "post_settle_max_tilt_deg": post_max, "max_tilt_deg": max([r["tilt"] for r in rows]+[final_tilt]),
              "min_height_m": min([r["height"] for r in rows]+[final.extra["height"]]),
              "post_mean_vx_m_s": measured_vx, "post_velocity_rmse_m_s": vx_rmse,
              "post_mean_planar_speed_m_s": speed_mean,
              "command_vx_m_s": args.speed if args.scenario == "walking" else 0.,
              "events": events, "scope": "One deterministic flat-ground MuJoCo trial; no external support forces." +
              (" H1 board collision removed for this flat-ground demo; preset motion-frame-0 spawn." if args.robot == "h1" else " Preset nominal standing spawn."),
              "criteria": {"post_seconds": args.post_seconds, "settle_allowance_seconds": 1., "max_post_tilt_deg": 15., "fall_tilt_deg": 45.,
                           "max_standing_mean_planar_speed_m_s": .15,
                           "max_walking_mean_vx_error_m_s": max(.15, .5*args.speed)}}
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    manifest = {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "scene": str(sim.xml.relative_to(ROOT)),
                "source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT/"stridebridge").glob("*.py"))},
                "asset_manifest_sha256": hashlib.sha256(asset_manifest.read_bytes()).hexdigest()}
    import scipy
    manifest["scipy"] = scipy.__version__
    if args.robot == "g1":
        import onnxruntime
        manifest["onnxruntime"] = onnxruntime.__version__
    else:
        import torch
        manifest["torch"] = torch.__version__
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=["g1", "h1"], default="g1")
    parser.add_argument("--scenario", choices=["standing", "walking"], default="standing")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-start", type=float, default=2.8)
    parser.add_argument("--motion-end", type=float, default=24.)
    parser.add_argument("--blend", type=float, default=.4)
    parser.add_argument("--entry-gap", type=float)
    parser.add_argument("--exit-gap", type=float, default=2.)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--post-seconds", type=float, default=6.)
    parser.add_argument("--max-seconds", type=float, default=45.)
    parser.add_argument("--left-assist", choices=["coordinated", "off"], default="coordinated")
    args = parser.parse_args()
    if args.speed is None:
        args.speed = .6 if args.robot == "h1" else .3
    if args.entry_gap is None:
        args.entry_gap = 1. if args.robot == "h1" else 1.5
    numbers = [args.motion_start, args.motion_end, args.blend, args.entry_gap, args.exit_gap, args.speed, args.post_seconds, args.max_seconds]
    if not np.isfinite(numbers).all() or min(numbers[2:5]+numbers[6:]) <= 0 or args.post_seconds < 2 or args.speed < 0:
        parser.error("Expected finite positive durations/gaps, post-seconds >= 2, and speed >= 0")
    raise SystemExit(0 if run(args) else 1)


if __name__ == "__main__":
    main()
