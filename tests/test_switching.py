"""Regression checks for command ownership and transitions."""
import numpy as np
import pytest

from stridebridge.switcher import PolicyAdapter, PolicyCommand, PolicySwitcher, RobotState


class ConstantPolicy(PolicyAdapter):
    def __init__(self, name, value):
        super().__init__(name, 2)
        self.value = value
        self.permit = True
        self.executed = []

    def step(self, state, command_scale):
        return PolicyCommand(np.full(2, self.value), np.full(2, 20.), np.full(2, 2.))

    def ready(self, state, command):
        return self.permit

    def accept_executed_command(self, command):
        self.executed.append(command.q.copy())


def fixture():
    a, b = ConstantPolicy("locomotion", 0.), ConstantPolicy("motion", 1.)
    state = RobotState(0., np.zeros(2), np.zeros(2))
    switch = PolicySwitcher(a, b, a.name, .4, .02, stable_seconds=.1, max_target_gap=1.1)
    switch.reset(state)
    return a, b, state, switch


def test_gate_keeps_locomotion_in_control_until_stable():
    a, b, state, switch = fixture()
    b.permit = False
    assert switch.request_switch("motion")
    for _ in range(10):
        cmd, status = switch.step(state)
        np.testing.assert_array_equal(cmd.q, [0., 0.])
        assert status["pending"] == "motion"
    b.permit = True
    for _ in range(4):
        _, status = switch.step(state)
        assert status["pending"] == "motion"
    _, status = switch.step(state)
    assert status["transition_target"] == "motion"
    weights = [switch.step(state)[1]["weights"]["motion"] for _ in range(21)]
    assert all(x <= y for x, y in zip(weights, weights[1:]))
    assert weights[-1] == 1.
    assert switch.active == "motion"
    np.testing.assert_array_equal(a.executed, b.executed)


def test_reset_cancels_old_pending_and_transition_state():
    _, _, state, switch = fixture()
    switch.request_switch("motion")
    for _ in range(10):
        switch.step(state)
    switch.reset(state)
    assert switch.active == "locomotion"
    assert switch.pending is None and switch.transition_target is None
    assert switch.request_switch("motion")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_feedback_is_rejected(bad):
    _, _, state, switch = fixture()
    state.q[0] = bad
    with pytest.raises(ValueError, match="feedback"):
        switch.step(state)


def test_negative_pd_gains_are_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        PolicyCommand(np.zeros(2), np.array([-1., 10.]), np.ones(2)).validate(2)


@pytest.mark.parametrize("dt", [0., -1., float("nan")])
def test_invalid_control_clock_is_rejected(dt):
    with pytest.raises(ValueError, match="timing"):
        PolicySwitcher(ConstantPolicy("a", 0), ConstantPolicy("b", 1), "a", .4, dt)
