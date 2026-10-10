"""Watchdog regressions independent of DDS sockets and the optional SDK."""
import numpy as np
import pytest

from stridebridge.dds import Feedback
from stridebridge.switcher import RobotState


def state(tick):
    return RobotState(tick/1000., np.zeros(2), np.zeros(2), {"tick": tick})


def test_frozen_tick_cannot_be_kept_alive_by_repeated_packets():
    feedback = Feedback(.1)
    feedback.update(state(1000), 1.)
    feedback.update(state(1000), 1.09)
    with pytest.raises(RuntimeError, match="Stale DDS"):
        feedback.read(1.11)


def test_packet_loss_expires_feedback_and_old_packets_cannot_refresh_it():
    feedback = Feedback(.1)
    feedback.update(state(1000), 1.)
    with pytest.raises(ValueError, match="backwards"):
        feedback.update(state(998), 1.09)
    with pytest.raises(RuntimeError, match="Stale DDS"):
        feedback.read(1.11)


def test_tick_wrap_is_accepted_and_control_reads_own_their_arrays():
    feedback = Feedback(.1)
    feedback.update(state(2**32-2), 1.)
    feedback.update(state(0), 1.002)
    snapshot = feedback.read(1.004)
    snapshot.q[:] = 7
    assert feedback.read(1.004).extra["tick"] == 0
    np.testing.assert_array_equal(feedback.read(1.004).q, 0)
