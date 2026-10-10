"""Bounded trailing-left-leg residual shared by deployment and simulation.

The default profile is the coordinated overlay verified on 2026-10-02.
Its arithmetic matches the frozen simulation wrapper. It changes position
targets only; policy observations, raw-action history and PD gains are owned
by the deployment controller.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np


TESTED_MOTOR_INDICES = (7, 3, 4, 5, 10, 8, 0, 1, 2, 11, 6, 16, 17, 18, 19, 12, 13, 14, 15)
TESTED_POLICY_SHA256 = "cfc0eba42b007df00e5ab1f99bb86053dd7256544373f47747b1cb6d6d8b5e47"
TESTED_MOTION_SHA256 = "23284b3920912df50a263f85fb5ccde21104709f8bfc03ae2ba3afaf41e08edc"


@dataclass(frozen=True)
class AssistSettings:
    mode: str = "coordinated"
    knee_cap: float = 2.02
    max_delta: float = .20
    margin: float = .03
    hip_delta: float = .06
    toe_up: float = .10
    start_frame: int = 90
    peak_frame: int = 98
    hold_frame: int = 103
    end_frame: int = 114


def add_assist_arguments(parser):
    parser.add_argument("--left-assist-mode", choices=("off", "knee", "knee_ankle", "coordinated"),
                        default="coordinated", help="trailing-left-leg position assist; default coordinated, off disables")
    for field, default in asdict(AssistSettings()).items():
        if field != "mode":
            parser.add_argument("--left-assist-" + field.replace("_", "-"),
                                type=int if field.endswith("frame") else float, default=default)


def settings_from_options(options):
    defaults = asdict(AssistSettings())
    return AssistSettings(**{name: getattr(options, "left_assist_" + name, default)
                             for name, default in defaults.items()})


def validate_assist_settings(settings, motion_clock="owned"):
    values = list(asdict(settings).values())[1:]
    if (settings.mode not in ("off", "knee", "knee_ankle", "coordinated")
            or not np.isfinite(values).all() or settings.margin <= 0
            or not 0 < settings.max_delta <= .30 or not 0 < settings.knee_cap < 2.05
            or not 0 <= settings.hip_delta <= .10 or not 0 <= settings.toe_up <= .15
            or not 0 <= settings.start_frame < settings.peak_frame <= settings.hold_frame < settings.end_frame < 154):
        raise ValueError("Invalid bounded left-leg assist settings")
    if settings.mode != "off" and motion_clock != "owned":
        raise ValueError("Left-leg assist requires --motion-clock owned; use --left-assist-mode off for other clocks")


def validate_assist_contract(settings, cfg, motion):
    if settings.mode == "off":
        return
    control_dt = float(cfg.control_dt)
    if (tuple(cfg.joint2motor_idx) != TESTED_MOTOR_INDICES
            or np.asarray(motion["joint_pos"]).shape != (154, 19)
            or not np.isfinite(control_dt) or abs(control_dt - .02) > 1e-9):
        raise ValueError("Left-leg assist requires the tested 19-joint, 154-frame, 50 Hz contract")
    lower = np.asarray(cfg.joint_limit_lower)
    upper = np.asarray(cfg.joint_limit_upper)
    default = np.asarray(cfg.default_angles)
    if (any(value.shape != (19,) or not np.isfinite(value).all()
            for value in (lower, upper, default))
            or np.any(lower >= upper)
            or not np.allclose(default, motion["joint_pos"][0], rtol=0., atol=1e-5)):
        raise ValueError("Left-leg assist requires valid 19-joint limits and the tested frame-0 default pose")
    for name, expected in (("policy_path", TESTED_POLICY_SHA256), ("motion_file", TESTED_MOTION_SHA256)):
        actual = hashlib.sha256(Path(getattr(cfg, name)).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Left-leg assist requires the tested model_30000 policy and motion; "
                             "use --left-assist-mode off with other assets")


def smoothstep(value):
    value = float(np.clip(value, 0., 1.))
    return value * value * (3. - 2. * value)


def envelope(frame, start, peak, hold, end):
    if frame <= start or frame >= end:
        return 0.
    if frame < peak:
        return smoothstep((frame - start) / (peak - start))
    if frame <= hold:
        return 1.
    return 1. - smoothstep((frame - hold) / (end - hold))


def assist_joint_targets(target, lower, upper, frame, settings):
    """Pure 19-joint target operation; retains margins and zero idle output."""
    result = np.asarray(target, dtype=np.float32).copy()
    if settings.mode == "off":
        return result, 0., 0.
    weight = envelope(frame, settings.start_frame, settings.peak_frame,
                      settings.hold_frame, settings.end_frame)
    hip, knee, ankle = 2, 3, 4  # native policy order; controller maps wire slots
    safe_knee = min(settings.knee_cap, float(upper[knee]) - settings.margin)
    knee_room = max(0., safe_knee - float(result[knee]))
    delta = min(settings.max_delta * weight, knee_room)
    if settings.mode == "knee_ankle":
        ankle_room = max(0., float(result[ankle]) - float(lower[ankle]) - settings.margin)
        delta = min(delta, ankle_room)
        result[ankle] -= delta
    elif settings.mode == "coordinated":
        hip_delta = -min(settings.hip_delta * weight, delta)
        ankle_delta = -(hip_delta + delta) - settings.toe_up * weight
        scale = 1.
        if ankle_delta < 0:
            ankle_room = max(0., float(result[ankle]) - float(lower[ankle]) - settings.margin)
            scale = min(scale, ankle_room / -ankle_delta)
        if hip_delta < 0:
            hip_room = max(0., float(result[hip]) - float(lower[hip]) - settings.margin)
            scale = min(scale, hip_room / -hip_delta)
        delta *= scale
        result[hip] += hip_delta * scale
        result[ankle] += ankle_delta * scale
    result[knee] += delta
    # The learned command is already joint-limited. Bound the overlay itself;
    # do not silently alter any other learned target or PD gain.
    if not (np.isfinite(result).all() and result[knee] <= float(upper[knee]) + 1e-6
            and result[ankle] >= float(lower[ankle]) - 1e-6):
        raise ValueError("Invalid left-leg residual target")
    return result, weight, delta
