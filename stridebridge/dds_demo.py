"""Run H1/G1 policy switching over actual SDK2 DDS in two local processes."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from .dds import DdsBackend
from .dds_sim import DdsSimulator, save_json
from .geometry import tilt
from .models import ROOT, build_policies, build_simulation
from .switcher import CommandLimiter, PolicySwitcher


def run_controller(args):
    walk, motion = build_policies(args)
    model = json.loads((args.output/"model.json").read_text())
    limiter = CommandLimiter(walk.motor_count, model["lower"], model["upper"])
    switch = PolicySwitcher(walk, motion, walk.name, args.blend, .02,
                            stable_seconds=.1, max_target_gap=args.entry_gap, limiter=limiter)
    backend = DdsBackend(args.robot, args.domain_id, args.packet_timeout)
    rows, events = [], []
    entered = requested_exit = False
    returned = None
    failure = None
    next_tick = time.monotonic()
    try:
        state = backend.wait_state()
        switch.reset(state)
        next_tick = time.monotonic()
        request_at = 6. if args.scenario == "walking" else 3.
        while True:
            began = time.monotonic()
            state = backend.read_state()
            t = state.timestamp
            if tilt(state.extra["quat"]) >= 45:
                raise RuntimeError("controller tilt threshold")
            if t >= args.max_seconds:
                raise RuntimeError("transition_timeout")
            requested = np.zeros(3, dtype=np.float32)
            if args.scenario == "walking" and (1 <= t < request_at-1.5 or (returned is not None and t >= returned+1.)):
                requested[0] = args.speed
            if not (switch.active == walk.name and switch.pending is None and switch.transition_target is None):
                requested[:] = 0
            if args.robot == "g1":
                walk.command[:] = requested
            else:
                walk.cmd[:] = requested/walk.max_cmd
            if t >= request_at and not entered:
                switch.request_switch(motion.name)
                events.append({"time": t, "event": "request:motion"})
                entered = True
            done = motion.done if args.robot == "g1" else motion.frame >= motion.motion_len
            if done and not requested_exit and switch.transition_target is None:
                switch.max_target_gap, switch.stable_seconds = args.exit_gap, .02
                switch.request_switch(walk.name)
                events.append({"time": t, "event": "request:locomotion"})
                requested_exit = True
            if args.robot == "h1":
                cfg = motion.cfg if switch.active == motion.name or switch.transition_target else SimpleNamespace(**walk.cfg)
                ids = motion.ids if cfg is motion.cfg else walk.ids
                limiter.q_lower[ids], limiter.q_upper[ids] = cfg.joint_limit_lower, cfg.joint_limit_upper
            command, status = switch.step(state)
            send = not (args.fault == "command-loss" and 3 <= t < 3.3)
            if send:
                backend.write_command(command, state)
            sent = time.monotonic()
            if status["event"]:
                events.append({"time": t, "event": status["event"]})
                print(round(t, 3), status["event"], flush=True)
                if requested_exit and status["active"] == walk.name:
                    returned = t
            rows.append({"time": t, "wall_time": began, "feedback_age": sent-state.extra["received_at"],
                         "tick_age": state.extra["tick_age"], "inference_seconds": sent-began,
                         "q": state.q, "dq": state.dq, "quat": state.extra["quat"], "omega": state.extra["omega"],
                         "command": requested.copy(), "target": command.q.copy(),
                         "motion_weight": status["weights"][motion.name], "sent": send,
                         "motion_time": motion.time if args.robot == "g1" else min(motion.frame, motion.motion_len)*.02})
            if returned is not None and t-returned >= args.post_seconds:
                break
            next_tick += .02
            delay = next_tick-time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if (args.output/"simulator_result.json").exists():
                raise RuntimeError("simulator ended before controller completion")
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)
    finally:
        if rows:
            np.savez_compressed(args.output/"controller.npz", **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})
        result = {"pid": os.getpid(), "failure": failure, "returned_at_s": returned, "events": events,
                  "states_received": backend.state_count, "invalid_states": backend.bad_states,
                  "commands_sent": backend.command_count, "policy_steps": len(rows)}
        save_json(args.output/"controller_result.json", result)
        print(json.dumps(result), flush=True)
        backend.close()
    return failure is None


def summarize(args, processes):
    results = {}
    for role in ("simulator", "controller"):
        path = args.output/(role+"_result.json")
        results[role] = json.loads(path.read_text()) if path.exists() else {"failure": "process produced no result"}
    sim, controller = results["simulator"], results["controller"]
    report = {"robot": args.robot, "scenario": args.scenario, "transport": "Unitree SDK2 DDS",
              "network_interface": "lo", "domain_id": args.domain_id,
              "success": False, "failure": sim["failure"] or controller["failure"],
              "fault_injection": args.fault, "events": controller.get("events", []),
              "processes": results, "exit_codes": processes,
              "scope": "Real-time MuJoCo with asynchronous DDS feedback/commands in separate processes. "
                       "Preset initial pose; flat ground; no support forces; no physical robot."}
    if (args.output/"simulator.npz").exists() and (args.output/"controller.npz").exists():
        with np.load(args.output/"simulator.npz") as source:
            log = {k: source[k] for k in source.files}
        with np.load(args.output/"controller.npz") as source:
            control = {k: source[k] for k in source.files}
        # Post-process recordings only. No simulator truth feeds policy inference.
        indices = np.clip(np.searchsorted(control["time"], log["time"], side="right")-1, 0, len(control["time"])-1)
        for key in ("command", "motion_weight", "motion_time", "feedback_age"):
            log[key] = control[key][indices]
        np.savez_compressed(args.output/"rollout.npz", **log)
        returned = controller.get("returned_at_s")
        post = np.zeros(len(log["time"]), dtype=bool) if returned is None else log["time"] >= returned+1.
        post_duration = 0. if returned is None else float(log["time"][-1]-returned)
        speed = float(np.mean(log["velocity"][post, 0])) if post.any() else None
        planar = float(np.mean(np.linalg.norm(log["velocity"][post, :2], axis=1))) if post.any() else None
        post_tilt = float(np.max(log["tilt"][post])) if post.any() else None
        wall, duration = sim.get("wall_seconds", 0.), sim.get("simulation_seconds", 0.)
        ratio = duration/wall if wall else 0.
        good_speed = post.any() and (planar < .15 if args.scenario == "standing" else abs(speed-args.speed) <= max(.15, args.speed*.5))
        report.update({"duration_s": float(log["time"][-1]+.02), "post_transition_duration_s": post_duration,
                       "post_settle_max_tilt_deg": post_tilt, "max_tilt_deg": float(np.max(log["tilt"])),
                       "min_height_m": float(np.min(log["height"])), "post_mean_vx_m_s": speed,
                       "post_mean_planar_speed_m_s": planar,
                       "command_vx_m_s": args.speed if args.scenario == "walking" else 0.,
                       "real_time_factor": ratio,
                       "feedback_age_p99_ms": float(np.percentile(control["feedback_age"], 99)*1000),
                       "max_feedback_age_ms": float(np.max(control["feedback_age"])*1000),
                       "max_command_age_ms": float(np.max(log["command_age"])*1000),
                       "inference_p99_ms": float(np.percentile(control["inference_seconds"], 99)*1000),
                       "policy_period_p99_ms": float(np.percentile(np.diff(control["wall_time"]), 99)*1000) if len(control["time"]) > 1 else None})
        report["success"] = bool(not report["failure"] and all(code == 0 for code in processes.values())
                                  and args.fault == "none" and good_speed and post_tilt < 15
                                  and post_duration >= args.post_seconds-.06 and .95 <= ratio <= 1.05
                                  and sim.get("invalid_commands") == 0 and controller.get("invalid_states") == 0
                                  and sim.get("commands_received") == controller.get("commands_sent")
                                  and sim.get("command_watchdog_at_s") is None)
    if not report["success"] and report["failure"] is None:
        report["failure"] = "acceptance_criteria"
    save_json(args.output/"metrics.json", report)
    return report


def run(args):
    if args.output.exists():
        raise FileExistsError("Choose a fresh DDS run directory: " + str(args.output))
    asset_manifest = ROOT/("assets/manifest.json" if args.robot == "g1" else "assets/local/h1/manifest.json")
    assets = json.loads(asset_manifest.read_text())
    groups = [(ROOT/"assets/public"/s["name"], s["files"]) for s in assets["sources"]] if args.robot == "g1" else [(ROOT/"assets/local/h1", assets)]
    for base, files in groups:
        for item in files:
            p = base/item["path"]
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Asset checksum mismatch: " + str(p))
    sim = build_simulation(args)
    # Static model limits only; live robot feedback never travels through files.
    args.output.mkdir(parents=True)
    # Unused H1 slot 9 has no joint limit; keep its target at zero in valid JSON.
    save_json(args.output/"model.json", {"lower": np.where(np.isfinite(sim.lower), sim.lower, 0.).tolist(),
                                         "upper": np.where(np.isfinite(sim.upper), sim.upper, 0.).tolist()})
    import mujoco
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    try:
        arguments["output"] = str(args.output.relative_to(ROOT))
    except ValueError:
        arguments["output"] = args.output.name
    manifest = {"args": arguments,
                "scene": str(sim.xml.relative_to(ROOT)), "transport": "dds", "network_interface": "lo",
                "physics_hz": 500, "feedback_hz": 500, "policy_hz": 50,
                "python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": np.__version__,
                "unitree_sdk2py": importlib.metadata.version("unitree_sdk2py"),
                "cyclonedds": importlib.metadata.version("cyclonedds"),
                "asset_manifest_sha256": hashlib.sha256(asset_manifest.read_bytes()).hexdigest(),
                "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT/"stridebridge").glob("*.py"))}}
    import onnxruntime
    import scipy
    manifest.update({"onnxruntime": onnxruntime.__version__, "scipy": scipy.__version__})
    import unitree_sdk2py
    sdk = Path(unitree_sdk2py.__file__).parent
    manifest["sdk_source_hashes"] = {
        name: hashlib.sha256((sdk/name).read_bytes()).hexdigest()
        for name in ("core/channel.py", "idl/default.py", "utils/crc.py")}
    if args.robot == "h1":
        import torch
        manifest["torch"] = torch.__version__
    report_criteria = {"post_seconds": args.post_seconds, "settle_allowance_seconds": 1.,
                       "max_post_tilt_deg": 15., "fall_tilt_deg": 45.,
                       "min_base_height_m": .45 if args.robot == "g1" else .65,
                       "max_standing_mean_planar_speed_m_s": .15,
                       "max_walking_mean_vx_error_m_s": max(.15, args.speed*.5),
                       "real_time_factor_range": [.95, 1.05], "packet_timeout_s": args.packet_timeout}
    manifest["criteria"] = report_criteria
    save_json(args.output/"manifest.json", manifest)
    lockpath = Path("/tmp")/f"stridebridge-dds-{os.getuid()}-{args.domain_id}.lock"
    children, logs = {}, []
    with lockpath.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for role in ("simulator", "controller"):
                log = (args.output/(role+".log")).open("w")
                logs.append(log)
                command = [sys.executable, "-u", "-m", "stridebridge.dds_demo", "--role", role]
                for key, value in vars(args).items():
                    if key != "role":
                        command.extend(["--"+key.replace("_", "-"), str(value)])
                children[role] = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+args.max_seconds+45
            while any(p.poll() is None for p in children.values()) and time.monotonic() < deadline:
                if any(p.poll() not in (None, 0) for p in children.values()):
                    # Give the other process time to record its watchdog/failure.
                    time.sleep(.3)
                    break
                time.sleep(.05)
        finally:
            for process in children.values():
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()
    report = summarize(args, {k: v.returncode for k, v in children.items()})
    print(json.dumps(report, indent=2))
    return report["success"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=("h1", "g1"), required=True)
    p.add_argument("--scenario", choices=("standing", "walking"), default="walking")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--domain-id", type=int)
    p.add_argument("--motion-start", type=float, default=2.8)
    p.add_argument("--motion-end", type=float, default=24.)
    p.add_argument("--blend", type=float, default=.4)
    p.add_argument("--entry-gap", type=float)
    p.add_argument("--exit-gap", type=float, default=2.)
    p.add_argument("--speed", type=float)
    p.add_argument("--post-seconds", type=float, default=6.)
    p.add_argument("--max-seconds", type=float, default=45.)
    p.add_argument("--packet-timeout", type=float, default=.1)
    p.add_argument("--left-assist", choices=("coordinated", "off"), default="coordinated")
    p.add_argument("--fault", choices=("none", "command-loss", "state-loss"), default="none",
                   help="Diagnostic packet interruption at t=3 s; a fault run should fail")
    p.add_argument("--role", choices=("run", "simulator", "controller"), default="run", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.domain_id is None:
        args.domain_id = 181 if args.robot == "h1" else 182
    if args.speed is None:
        args.speed = .6 if args.robot == "h1" else .3
    if args.entry_gap is None:
        args.entry_gap = 1. if args.robot == "h1" else 1.5
    numbers = [args.motion_start, args.motion_end, args.blend, args.entry_gap, args.exit_gap, args.speed,
               args.post_seconds, args.max_seconds, args.packet_timeout]
    if not np.isfinite(numbers).all() or min(numbers[2:5]+numbers[6:]) <= 0 or args.speed < 0 or args.post_seconds < 2:
        p.error("Expected finite positive timings/gaps, nonnegative speed, and post-seconds >= 2")
    if not 1 <= args.domain_id <= 232:
        p.error("Use an isolated DDS domain in [1, 232]; interface is always lo")
    if args.role == "controller":
        success = run_controller(args)
    elif args.role == "simulator":
        success = DdsSimulator(args).run()
    else:
        success = run(args)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
