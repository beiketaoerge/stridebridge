#!/usr/bin/env python3
"""Offline operator regressions against actual controller methods.

AST loading deliberately excludes module imports, main(), constructors and SDK
initialization. The actual PolicySwitcher and DdsBackend methods execute against
synthetic feedback, a virtual clock and a recorder of all 20 motor fields. No
DDS, model inference, networking, robot access or realtime sleeps occur.

Run: /usr/bin/python3 -B tests/test_operator_flow.py
Optional: --controller PATH --report PATH (JSON, only when explicitly supplied).
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = ROOT / "deploy_real_h1_motion_104_h1switch_0927.py"
FIELDS = ("q", "dq", "kp", "kd", "tau")
KeyMap = SimpleNamespace(start=0, select=1, A=2, B=3, R1=4)


def load_switcher():
    spec = importlib.util.spec_from_file_location("offline_operator_switcher", ROOT / "h1_policy_switcher.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SWITCH = load_switcher()


class VirtualClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = 0
        self.reads = 0
        self.on_sleep = None
        self.owner = None
        self.refresh = True

    def monotonic(self):
        self.reads += 1
        if self.reads > 10000:
            raise TimeoutError("Offline test watchdog: controller did not leave its wait")
        return self.now

    def sleep(self, delay):
        if delay < 0:
            raise AssertionError("Controller tried to sleep for a negative duration")
        self.now += float(delay)
        self.sleeps += 1
        if self.sleeps > 300:
            raise TimeoutError("Offline test watchdog: controller did not leave its wait")
        if self.refresh and self.owner is not None:
            self.owner.msg.tick += 1
            self.owner.feedback = (self.owner.msg, self.now)
        if self.on_sleep is not None:
            self.on_sleep(self.sleeps)


def zero_fields(cmd):
    for motor in cmd.motor_cmd:
        for field in FIELDS:
            setattr(motor, field, 0.0)


def damping_fields(cmd):
    zero_fields(cmd)
    for motor in cmd.motor_cmd:
        motor.kd = 8.0


class ForbiddenBaseController:
    def zero_torque_state(self):
        raise AssertionError("Startup must override the base wait, which does not validate feedback or Select")


def load_controller(clock):
    source = CONTROLLER_PATH.read_text()
    tree = ast.parse(source, filename=str(CONTROLLER_PATH))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in {"OperatorStop", "StartupCancelled", "DdsBackend", "H1SwitchController", "tilt"}]
    namespace = {
        "__name__": "offline_operator_controller",
        "__file__": str(CONTROLLER_PATH),
        "np": np,
        "time": clock,
        "KeyMap": KeyMap,
        "base": SimpleNamespace(
            Controller=ForbiddenBaseController,
            create_zero_cmd=zero_fields,
            create_damping_cmd=damping_fields,
            transform_imu_data=lambda yaw, dyaw, quat, omega: (np.asarray(quat), np.asarray(omega))),
        "OPTIONS": SimpleNamespace(net="offline", sim_prepare=False, stand_seconds=3.0,
                                   entry_timeout=10.0, entry_seconds=0.4, exit_seconds=0.4,
                                   exit_max_target_gap=2.0, exit_stable_seconds=0.02),
    }
    for name in ("RobotBackend", "RobotState", "PolicyCommand", "PolicySwitcher", "CommandLimiter"):
        namespace[name] = getattr(SWITCH, name)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(CONTROLLER_PATH), "exec"), namespace)
    return namespace


class SyntheticPolicy(SWITCH.PolicyAdapter):
    def __init__(self, name):
        super().__init__(name, 20)
        self.ids = list(range(19))
        self.default = np.full(19, 0.3, dtype=np.float32)
        self.kp = np.full(20, 100.0, dtype=np.float32)
        self.kd = np.full(20, 2.0, dtype=np.float32)
        self.obs = np.zeros(70 if name == "stand" else 104, dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)
        self.calls = 0
        self.on_step = None
        self.frame = self.sample_frame = 0
        self.motion_len = 154
        self.raw_action = np.zeros(19, dtype=np.float32)
        self.clipped_action = self.raw_action.copy()
        self.raw_target = self.default.copy()
        # Actual run() now records the deployment-integrated residual. These
        # fixtures keep it zero so existing operator tests still isolate stops.
        self.assist_weight = self.assist_knee_delta = 0.0
        self.assist_base_target = self.raw_target.copy()
        self.assist_target = self.raw_target.copy()
        self.assist_settings = SimpleNamespace(mode="coordinated")
        self.last_command = SWITCH.PolicyCommand(np.zeros(20, dtype=np.float32), self.kp, self.kd)

    def step(self, state, command_scale):
        self.calls += 1
        if self.on_step is not None:
            self.on_step()
        self.last_command = SWITCH.PolicyCommand(np.full(20, 0.3, dtype=np.float32), self.kp.copy(), self.kd.copy())
        return self.last_command

    def ready(self, state, command):
        # Keep crossing requests pending: tests can distinguish cancel from blend.
        return self.name == "stand"


class Harness:
    def __init__(self):
        self.clock = VirtualClock()
        self.namespace = load_controller(self.clock)
        self.owner = object.__new__(self.namespace["H1SwitchController"])
        c = self.owner
        c.config = SimpleNamespace(control_dt=0.02, torso_idx=6, joint2motor_idx=list(range(19)))
        c.remote_controller = SimpleNamespace(button=[0] * 16, lx=0.0, ly=0.0, rx=0.0)
        c.sim_auto_start = c.sim_prepare = False
        c.startup_in_progress = False
        c._startup_manual_restart = False
        c.run_steps = None
        c.motion_len = 154
        c.msg = SimpleNamespace(
            tick=1,
            imu_state=SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[0.0] * 3),
            motor_state=[SimpleNamespace(q=0.1, dq=0.0) for _ in range(20)])
        c.feedback = (c.msg, 0.0)
        c.low_cmd = SimpleNamespace(motor_cmd=[SimpleNamespace(q=7.0, dq=6.0, kp=5.0, kd=4.0, tau=3.0)
                                                   for _ in range(20)])
        c.stand, c.cross = SyntheticPolicy("stand"), SyntheticPolicy("cross")
        lower, upper = np.full(20, -3.0, dtype=np.float32), np.full(20, 3.0, dtype=np.float32)
        c.lim_cross = c.lim_walk = (lower, upper)
        c.switcher = SWITCH.PolicySwitcher(c.stand, c.cross, "stand", 0.4, 0.02,
                                         stable_seconds=0.1,
                                         limiter=SWITCH.CommandLimiter(20, lower, upper))
        c.events = []
        c.counter = 0
        c.crossing_start_row = -1
        c.qj = np.zeros(19, dtype=np.float32)
        c.dqj = c.qj.copy()
        for name in ("weights", "stand_obs", "kp", "kd", "age", "gap", "cmd",
                     "assist_weight", "assist_knee_delta", "assist_base_target", "assist_target"):
            setattr(c, "trace_" + name, [])
        c._append_rollout_sample = lambda *args: None
        c._save_rollout = lambda: None
        self.messages = []
        self.output = io.StringIO()
        self.on_crc = None
        def crc(cmd):
            if self.on_crc is not None:
                self.on_crc()
            return 123
        self.namespace["base"].CRC = lambda: SimpleNamespace(Crc=crc)
        c.lowcmd_publisher_ = SimpleNamespace(Write=self.record)
        if not hasattr(c, "send_cmd"):
            # Before-fix snapshots inherit SDK transport; that dependency is
            # replaced only at the transport boundary for baseline comparison.
            c.send_cmd = self.transport_stub
        c.backend = self.namespace["DdsBackend"](c)
        self.clock.owner = c

    def transport_stub(self, cmd, *, before_write=None):
        if before_write is not None:
            before_write()
        self.record(cmd)

    def record(self, cmd):
        motors = np.asarray([[getattr(m, f) for f in FIELDS] for m in cmd.motor_cmd], dtype=float)
        self.messages.append({"time": self.clock.now, "motors": motors,
                              "buttons": self.owner.remote_controller.button.copy()})

    def button(self, name, value=1):
        self.owner.remote_controller.button[getattr(KeyMap, name)] = value

    def call(self, name):
        with contextlib.redirect_stdout(self.output):
            if name in ("move_to_motion_start", "default_pos_state"):
                self.owner.startup_in_progress = True
                try:
                    getattr(self.owner, name)()
                finally:
                    self.owner.startup_in_progress = False
            else:
                getattr(self.owner, name)()

    def zero_messages(self):
        return [m for m in self.messages if np.all(m["motors"] == 0)]

    def position_messages(self):
        return [m for m in self.messages if np.any(m["motors"][:, 2] > 0)]

    def damping_messages(self):
        return [m for m in self.messages if np.all(m["motors"][:, 2] == 0)
                and np.all(m["motors"][:, 3] > 0)]


class OperatorFlowTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()

    def assert_cancelled(self, fn):
        cls = self.h.namespace.get("StartupCancelled")
        self.assertIsNotNone(cls, "Controller must distinguish startup cancellation from a fault")
        with self.assertRaises(cls):
            fn()

    def test_zero_torque_clears_every_field_on_every_motor(self):
        self.h.owner.backend.zero_torque()
        self.assertEqual(len(self.h.messages), 1)
        np.testing.assert_array_equal(self.h.messages[0]["motors"], np.zeros((20, 5)))

    def test_start_and_select_requires_release_then_fresh_start(self):
        h = self.h
        h.button("start")
        h.button("select")
        def buttons(n):
            if n == 1:
                h.button("select", 0)  # Holding Start after release must not launch.
            elif n == 2:
                h.button("start", 0)
            elif n == 3:
                h.button("start")
        h.clock.on_sleep = buttons
        h.call("zero_torque_state")
        self.assertGreaterEqual(h.clock.sleeps, 3)
        self.assertTrue(h.messages)
        self.assertEqual(len(h.zero_messages()), len(h.messages))

    def test_start_wait_rejects_lost_feedback(self):
        self.h.owner.feedback = None
        with self.assertRaises(RuntimeError):
            self.h.call("zero_torque_state")

    def test_start_wait_rejects_stale_feedback(self):
        self.h.owner.feedback = (self.h.owner.msg, -0.2)
        with self.assertRaises(RuntimeError):
            self.h.call("zero_torque_state")
        self.assertFalse(self.h.position_messages())

    def test_start_wait_rejects_invalid_raw_imu(self):
        self.h.owner.msg.imu_state.quaternion = [0.0] * 4
        with self.assertRaises(RuntimeError):
            self.h.call("zero_torque_state")

    def test_select_on_fifth_ramp_tick_emits_no_sixth_position(self):
        h = self.h
        h.clock.on_sleep = lambda n: h.button("select") if n == 5 else None
        self.assert_cancelled(lambda: h.call("move_to_motion_start"))
        self.assertEqual(len(h.position_messages()), 5)
        self.assertFalse(h.damping_messages())

    def test_select_during_stance_hold_emits_no_more_position(self):
        h = self.h
        h.clock.on_sleep = lambda n: h.button("select") if n == 2 else None
        self.assert_cancelled(lambda: h.call("default_pos_state"))
        self.assertEqual(len(h.position_messages()), 2)
        self.assertFalse(h.damping_messages())

    def test_startup_select_returns_to_zero_and_demands_fresh_start(self):
        h = self.h
        def buttons(n):
            if n == 1:
                h.button("start")
            elif n == 4:
                h.button("select")  # Cancel while the first ramp is moving.
            elif n == 5:
                h.button("select", 0)  # Start remains held for this tick.
            elif n == 6:
                h.button("start", 0)
            elif n == 7:
                h.button("start")
            elif n == 108:
                h.button("A")
        h.clock.on_sleep = buttons
        h.call("startup")
        interval = [m for m in h.messages if 0.08 - 1e-9 <= m["time"] < 0.14 - 1e-9]
        self.assertTrue(interval, "Cancellation must publish zero torque before waiting again")
        self.assertTrue(all(np.all(m["motors"] == 0) for m in interval))
        self.assertFalse(h.damping_messages(), "An operator cancellation in startup must use zero torque")
        self.assertGreater(len(h.position_messages()), 100)

    def test_ramp_stops_on_feedback_loss(self):
        h = self.h
        def disconnect(n):
            if n == 3:
                h.clock.refresh = False
        h.clock.on_sleep = disconnect
        with self.assertRaises(RuntimeError):
            h.call("move_to_motion_start")
        self.assertLess(len(h.position_messages()), 15)

    def test_ramp_rejects_invalid_feedback_before_position(self):
        self.h.owner.msg.imu_state.quaternion = [float("nan"), 0.0, 0.0, 0.0]
        with self.assertRaises(RuntimeError):
            self.h.call("move_to_motion_start")
        self.assertFalse(self.h.position_messages())

    def test_run_select_already_pressed_never_infers_or_publishes_position(self):
        h = self.h
        h.button("select")
        h.call("run")
        self.assertEqual(h.owner.stand.calls, 0)
        self.assertEqual(h.owner.cross.calls, 0)
        self.assertFalse(h.position_messages())
        self.assertTrue(h.damping_messages())

    def test_run_select_arriving_during_inference_never_publishes_position(self):
        h = self.h
        h.owner.stand.on_step = lambda: h.button("select")
        h.call("run")
        self.assertEqual(h.owner.stand.calls, 1)
        self.assertFalse(h.position_messages())
        self.assertTrue(h.damping_messages())

    def test_cancel_beats_same_tick_r1_a_b_request(self):
        h = self.h
        h.owner.switcher.request_switch("cross")
        def buttons(n):
            if n == 1:
                for name in ("R1", "A", "B"):
                    h.button(name)
            elif n == 2:
                h.button("select")
        h.clock.on_sleep = buttons
        h.call("run")
        names = [e["event"] for e in h.owner.events]
        self.assertIn("cancel:pending", names)
        self.assertNotIn("request:cross", names)
        self.assertIsNone(h.owner.switcher.pending)

    def test_holding_b_blocks_a_later_cross_request(self):
        h = self.h
        h.button("B")
        h.button("R1")
        def buttons(n):
            if n == 1:
                h.button("A")
            elif n == 2:
                h.button("select")
        h.clock.on_sleep = buttons
        h.call("run")
        self.assertNotIn("request:cross", [e["event"] for e in h.owner.events])
        self.assertIsNone(h.owner.switcher.pending)

    def test_position_publish_rechecks_feedback_age(self):
        h = self.h
        h.owner.feedback = (h.owner.msg, -0.2)
        command = SWITCH.PolicyCommand(np.full(20, 0.3, dtype=np.float32),
                                       np.full(20, 100.0, dtype=np.float32),
                                       np.full(20, 2.0, dtype=np.float32))
        with self.assertRaises(RuntimeError):
            h.owner.backend.write_command(command)
        self.assertFalse(h.position_messages())

    def test_inference_stall_rejects_position_and_damps(self):
        h = self.h
        def stalled_inference():
            h.clock.now += 0.2  # No incoming LowState while inference is stalled.
        h.owner.stand.on_step = stalled_inference
        with self.assertRaises(RuntimeError):
            h.call("run")
        self.assertFalse(h.position_messages())
        self.assertTrue(h.damping_messages())

    def test_fresh_feedback_does_not_hide_stale_inference_snapshot(self):
        h = self.h
        def stalled_inference_with_new_state():
            h.clock.now += 0.2
            h.owner.msg.tick += 1
            h.owner.feedback = (h.owner.msg, h.clock.now)
        h.owner.stand.on_step = stalled_inference_with_new_state
        with self.assertRaisesRegex(RuntimeError, "older|old|age|fresh"):
            h.call("run")
        self.assertFalse(h.position_messages())
        self.assertTrue(h.damping_messages())

    def test_held_r1_a_from_stance_does_not_request_crossing(self):
        h = self.h
        h.button("R1")
        h.button("A")
        h.clock.on_sleep = lambda n: h.button("select") if n == 1 else None
        h.call("run")
        self.assertNotIn("request:cross", [e["event"] for e in h.owner.events])

    def test_held_a_release_and_new_r1_a_press_can_request_crossing(self):
        h = self.h
        h.button("R1")
        h.button("A")
        def buttons(n):
            if n == 1:
                h.button("A", 0)
            elif n == 2:
                h.button("A")
            elif n == 3:
                h.button("select")
        h.clock.on_sleep = buttons
        h.call("run")
        self.assertEqual([e["event"] for e in h.owner.events].count("request:cross"), 1)

    def test_run_select_arrives_inside_final_feedback_read(self):
        h = self.h
        original = h.owner.backend.read_state
        reads = 0
        def read_then_select():
            nonlocal reads
            state = original()
            reads += 1
            if reads == 3:  # Initial state, loop snapshot, then publication guard.
                h.button("select")
            return state
        h.owner.backend.read_state = read_then_select
        h.call("run")
        self.assertEqual(h.owner.stand.calls, 1)
        self.assertFalse(h.position_messages())
        self.assertTrue(h.damping_messages())

    def test_startup_select_arrives_inside_final_feedback_read(self):
        h = self.h
        original = h.owner.backend.read_state
        reads = 0
        def read_then_select():
            nonlocal reads
            state = original()
            reads += 1
            if reads == 3:
                h.button("select")
            return state
        h.owner.backend.read_state = read_then_select
        self.assert_cancelled(lambda: h.call("move_to_motion_start"))
        self.assertFalse(h.position_messages())
        self.assertFalse(h.damping_messages())

    def test_select_on_final_ramp_sleep_still_cancels_startup(self):
        h = self.h
        h.clock.on_sleep = lambda n: h.button("select") if n == 100 else None
        self.assert_cancelled(lambda: h.call("move_to_motion_start"))
        self.assertEqual(len(h.position_messages()), 100)
        self.assertFalse(h.damping_messages())

    def test_auto_start_cancellation_requires_manual_start_and_a(self):
        h = self.h
        h.owner.sim_auto_start = True
        h.button("start")
        h.button("select")
        def buttons(n):
            if n == 1:
                h.button("select", 0)
            elif n == 2:
                h.button("start", 0)
            elif n == 3:
                h.button("start")
            elif n == 105:
                h.button("A")
        h.clock.on_sleep = buttons
        h.call("startup")
        self.assertGreaterEqual(h.clock.sleeps, 105, "Auto-start must not bypass the manual A wait after cancellation")
        before_start = [m for m in h.messages if m["time"] < 0.06 - 1e-9]
        self.assertTrue(before_start)
        self.assertTrue(all(np.all(m["motors"] == 0) for m in before_start))
        self.assertFalse(h.damping_messages())

    def test_select_during_crc_never_reaches_final_position_publish(self):
        h = self.h
        self.assertIn("send_cmd", h.owner.__class__.__dict__, "Controller must guard the final publisher boundary")
        h.on_crc = lambda: h.button("select")
        h.call("run")
        self.assertEqual(h.owner.stand.calls, 1)
        self.assertFalse(h.position_messages())
        self.assertTrue(h.damping_messages())

    def test_startup_select_during_crc_never_publishes_position(self):
        h = self.h
        self.assertIn("send_cmd", h.owner.__class__.__dict__, "Controller must guard the final publisher boundary")
        h.on_crc = lambda: h.button("select")
        self.assert_cancelled(lambda: h.call("move_to_motion_start"))
        self.assertFalse(h.position_messages())
        self.assertFalse(h.damping_messages())


class JsonResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.outcomes.append({"test": test.id(), "status": "PASS"})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.outcomes.append({"test": test.id(), "status": "FAIL", "detail": self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.outcomes.append({"test": test.id(), "status": "ERROR", "detail": self._exc_info_to_string(err, test)})


def main():
    global CONTROLLER_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=CONTROLLER_PATH)
    parser.add_argument("--report", type=Path, help="Write a JSON report at this explicit path")
    args = parser.parse_args()
    CONTROLLER_PATH = args.controller.resolve()
    before_sha = hashlib.sha256(CONTROLLER_PATH.read_bytes()).hexdigest()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OperatorFlowTests)
    result = unittest.TextTestRunner(verbosity=2, resultclass=JsonResult).run(suite)
    after_sha = hashlib.sha256(CONTROLLER_PATH.read_bytes()).hexdigest()
    report = {"controller": str(CONTROLLER_PATH),
              "controller_sha256": after_sha,
              "source_unchanged_during_tests": before_sha == after_sha,
              "n_total": result.testsRun, "n_pass": sum(x["status"] == "PASS" for x in result.outcomes),
              "all_passed": result.wasSuccessful() and before_sha == after_sha, "offline": True,
              "coverage": "AST-loaded actual control methods; actual PolicySwitcher; virtual clock and feedback; all motor q/dq/kp/kd/tau recorded",
              "limitations": "No SDK/DDS, physical remote, motor response, real TorchScript inference, or stability verification",
              "baseline": {
                  "controller_sha256": "1cec814ab21ebeed6a81ce7096725ff3ddba246ede9e0501da7c1309a53b761c",
                  "assessment": str(ROOT.parent / "TASK_COMPLETION_ASSESSMENT_20261002.md"),
                  "prior_isolated_reproductions": [
                      "Select already pressed at run entry still allowed one position command before damping",
                      "Same-tick R1+A+B cancelled pending cross and then requested cross again"],
                  "same_suite_rerun": False,
                  "reason": "Original source was not retained; baseline refers only to the earlier recorded isolated reproductions"},
              "tests": result.outcomes}
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("controller_sha256", "n_total", "n_pass", "all_passed", "offline")}))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
