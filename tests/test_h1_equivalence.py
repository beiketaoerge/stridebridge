"""Compare migrated inference with the frozen latest controller without DDS."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from stridebridge import geometry, left_leg_assist
from stridebridge.switcher import PolicyAdapter, PolicyCommand

ROOT = Path(__file__).resolve().parents[1]


def test_h1_adapters_match_latest_controller_on_feedback_replay():
    torch = pytest.importorskip("torch")
    if not (ROOT/"assets/local/h1/manifest.json").exists():
        pytest.skip("Requires the local H1 model bundle")
    from stridebridge.demo import build_h1
    options = SimpleNamespace(left_assist="coordinated")
    sim, walk, motion = build_h1(options)
    source = ROOT/"examples/h1_latest/deploy_real_h1_motion_104_h1switch_0927.py"
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in {"Dwaq0816Adapter", "MotionAdapter"}]
    base_path = ROOT/"assets/local/h1/0927/deploy/deploy_real/deploy_real_h1_motion_104_realfix.py"
    base_tree = ast.parse(base_path.read_text())
    functions = [n for n in base_tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"quat_conj", "quat_mul", "quat_to_matrix", "yaw_quat"}]
    original_math = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(base_path), "exec"), original_math)
    scope = {"np": np, "torch": torch, "yaml": yaml, "Path": Path, "PolicyAdapter": PolicyAdapter,
             "PolicyCommand": PolicyCommand, "base": SimpleNamespace(**original_math), "tilt": geometry.tilt,
             "OPTIONS": motion.owner.options}
    for name in ["settings_from_options", "validate_assist_settings", "validate_assist_contract", "assist_joint_targets"]:
        scope[name] = getattr(left_leg_assist, name)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), scope)
    original_walk = scope["Dwaq0816Adapter"](ROOT/"assets/local/h1/0823/deploy/deploy_real/configs/h1_dwaq_0816_box3kg.yaml")
    original_motion = scope["MotionAdapter"](motion.owner)
    rng = np.random.RandomState(43)
    state = sim.read()
    for policy in [walk, motion, original_walk, original_motion]:
        policy.reset(state)
    for i in range(160):
        state.q += rng.normal(0., .0005, state.q.shape).astype(np.float32)
        state.dq[:] = rng.normal(0., .01, state.dq.shape).astype(np.float32)
        for a, b in [(walk, original_walk), (motion, original_motion)]:
            ca, cb = a.step(state, 1.), b.step(state, 1.)
            for attr in ["q", "kp", "kd"]:
                np.testing.assert_array_equal(getattr(ca, attr), getattr(cb, attr))
            np.testing.assert_array_equal(a.obs, b.obs)
            ca.metadata["blend_weight"] = cb.metadata["blend_weight"] = float(i >= 3)
            a.accept_executed_command(ca)
            b.accept_executed_command(cb)
