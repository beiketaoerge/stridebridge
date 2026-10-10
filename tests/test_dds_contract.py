"""Exercise real SDK IDL serialization, CRC and robot-specific motor/IMU contracts."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("unitree_sdk2py")
from stridebridge.dds import Wire
from stridebridge.switcher import PolicyCommand


@pytest.mark.parametrize("robot,count,slots", [("h1", 20, 20), ("g1", 29, 35)])
def test_real_idl_serialization_keeps_sdk_motor_order_and_disables_unused_slots(robot, count, slots):
    wire = Wire(robot)
    command = PolicyCommand(np.arange(count, dtype=np.float32)*.01, np.full(count, 20.), np.full(count, 2.))
    msg = wire.encode_command(command, mode_machine=5)
    msg = wire.command_type.deserialize(msg.serialize())
    assert len(msg.motor_cmd) == slots
    assert msg.motor_cmd[7].q == pytest.approx(.07)
    assert msg.motor_cmd[10].mode == 1
    for i in set(range(slots))-set(wire.enabled):
        m = msg.motor_cmd[i]
        assert (m.mode, m.q, m.dq, m.tau, m.kp, m.kd) == (0, 0., 0., 0., 0., 0.)
    decoded = wire.decode_command(msg, mode_machine=5)
    np.testing.assert_allclose(decoded.command.q[wire.enabled], command.q[wire.enabled])
    msg.motor_cmd[7].q += .1
    with pytest.raises(ValueError, match="CRC"):
        wire.decode_command(msg, mode_machine=5)


def test_g1_rejects_wrong_machine_or_parallel_motor_mode_even_with_valid_crc():
    wire = Wire("g1")
    msg = wire.encode_command(PolicyCommand(np.zeros(29), np.ones(29), np.ones(29)), mode_machine=5)
    with pytest.raises(ValueError, match="machine/PR"):
        wire.decode_command(msg, mode_machine=0)
    msg.mode_pr = 1
    msg.crc = wire.crc.Crc(msg)
    with pytest.raises(ValueError, match="machine/PR"):
        wire.decode_command(msg, mode_machine=5)


@pytest.mark.parametrize("field,value,index", [("kp", -1., 3), ("q", float("nan"), 3), ("tau", 2., 34)])
def test_crc_does_not_make_invalid_motor_commands_acceptable(field, value, index):
    wire = Wire("g1")
    msg = wire.encode_command(PolicyCommand(np.zeros(29), np.ones(29), np.ones(29)))
    setattr(msg.motor_cmd[index], field, value)
    msg.crc = wire.crc.Crc(msg)
    with pytest.raises(ValueError, match="invalid motor|unused motor"):
        wire.decode_command(msg)


@pytest.mark.parametrize("robot", ["h1", "g1"])
def test_imu_recovers_pelvis_at_nonzero_waist_rotation_and_angular_velocity(robot):
    wire = Wire(robot)
    msg = wire.new_state()
    pelvis = Rotation.from_euler("xyz", [.2, -.3, .4])
    omega = np.array([.4, -.2, .1])
    if robot == "h1":
        msg.head[:] = [0xFE, 0xEF]
        msg.level_flag = 0xFF
        msg.motor_state[6].q, msg.motor_state[6].dq = .7, .9
        waist = Rotation.from_euler("z", .7)
        imu = pelvis*waist
        imu_omega = waist.inv().apply(omega + [0, 0, .9])
    else:
        imu, imu_omega = pelvis, omega
    msg.imu_state.quaternion[:] = imu.as_quat()[[3, 0, 1, 2]].tolist()
    msg.imu_state.gyroscope[:] = imu_omega.tolist()
    msg.tick = 100
    msg.crc = wire.crc.Crc(msg)
    msg = wire.state_type.deserialize(msg.serialize())
    state = wire.decode_state(msg)
    np.testing.assert_allclose(state.extra["quat"], pelvis.as_quat()[[3, 0, 1, 2]], atol=1e-7)
    np.testing.assert_allclose(state.extra["omega"], omega, atol=1e-7)
    msg.imu_state.quaternion[:] = [0., 0., 0., 0.]
    msg.crc = wire.crc.Crc(msg)
    with pytest.raises(ValueError, match="IMU"):
        wire.decode_state(msg)
