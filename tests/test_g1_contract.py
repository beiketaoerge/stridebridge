"""Regressions for the action-history bug that caused G1 handback falls."""
import numpy as np
from stridebridge.g1 import G1Mimic, G1Policy
from stridebridge.switcher import PolicyCommand


def test_velocity_history_uses_executed_motor_targets_in_policy_order():
    policy = object.__new__(G1Policy)
    policy.ids = np.array([2, 0, 1])
    policy.offset = np.array([.1, .2, .3])
    policy.scale = np.array([.25, .5, 1.])
    policy.action = np.zeros(3)
    executed = PolicyCommand(np.array([1.2, 2.3, .6]), np.ones(3), np.ones(3))
    policy.accept_executed_command(executed)
    np.testing.assert_allclose(policy.action, [2., 2., 2.])


def test_reference_time_advances_only_with_full_motion_ownership():
    policy = object.__new__(G1Mimic)
    policy.time, policy.end, policy.dt = 2.8, 3., .02
    policy.started = False
    policy.raw_action = np.ones(29)
    policy.action = np.zeros(29)
    for weight in [0., 0., .2, .5, .9]:
        policy.accept_executed_command(PolicyCommand(np.zeros(29), np.ones(29), np.ones(29), {"blend_weight": weight}))
        assert policy.time == 2.8
    for _ in range(30):
        policy.accept_executed_command(PolicyCommand(np.zeros(29), np.ones(29), np.ones(29), {"blend_weight": 1.}))
    assert policy.time == 3. and policy.done
