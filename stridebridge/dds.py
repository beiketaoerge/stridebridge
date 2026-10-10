"""Unitree GO (H1) / HG (G1) wire contracts and loopback DDS feedback backend.

No DDS participant is opened on import. Demo connections are restricted to lo;
the simulator exchanges the standard rt/lowcmd and rt/lowstate SDK2 messages.
"""
from dataclasses import dataclass
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import quat_to_matrix
from .switcher import PolicyCommand, RobotState


@dataclass
class MotorPacket:
    command: PolicyCommand
    dq: np.ndarray
    tau: np.ndarray


class Wire:
    def __init__(self, robot):
        from unitree_sdk2py.idl import default
        from unitree_sdk2py.utils.crc import CRC
        if robot not in ("h1", "g1"):
            raise ValueError("Expected h1 or g1")
        self.robot = robot
        self.count = 20 if robot == "h1" else 29
        self.enabled = [i for i in range(self.count) if robot != "h1" or i != 9]
        family = "go" if robot == "h1" else "hg"
        self.new_command = getattr(default, f"unitree_{family}_msg_dds__LowCmd_")
        self.new_state = getattr(default, f"unitree_{family}_msg_dds__LowState_")
        self.command_type = type(self.new_command())
        self.state_type = type(self.new_state())
        self.crc = CRC()

    def motor_mode(self, index):
        if index not in self.enabled:
            return 0
        return 0x0A if self.robot == "h1" and index < 10 else 1

    def encode_command(self, command, mode_machine=0):
        command.validate(self.count)
        msg = self.new_command()
        if self.robot == "h1":
            msg.head[:] = [0xFE, 0xEF]
            msg.level_flag = 0xFF
        else:
            msg.mode_pr, msg.mode_machine = 0, int(mode_machine)
        for i, motor in enumerate(msg.motor_cmd):
            motor.mode = self.motor_mode(i)
            motor.q = motor.dq = motor.tau = motor.kp = motor.kd = 0.
            if i in self.enabled:
                motor.q, motor.kp, motor.kd = map(float, (command.q[i], command.kp[i], command.kd[i]))
        msg.crc = self.crc.Crc(msg)
        return msg

    def decode_command(self, msg, mode_machine=0):
        if not isinstance(msg, self.command_type) or msg.crc != self.crc.Crc(msg):
            raise ValueError("command CRC/type")
        if self.robot == "h1":
            if list(msg.head) != [0xFE, 0xEF] or msg.level_flag != 0xFF:
                raise ValueError("command header")
        elif msg.mode_pr != 0 or msg.mode_machine != mode_machine:
            raise ValueError("command machine/PR mode")
        for i, motor in enumerate(msg.motor_cmd):
            if motor.mode != self.motor_mode(i):
                raise ValueError("command motor mode")
            values = np.array([motor.q, motor.dq, motor.tau, motor.kp, motor.kd])
            if not np.isfinite(values).all() or min(motor.kp, motor.kd) < 0:
                raise ValueError("invalid motor command")
            if i not in self.enabled and np.any(values):
                raise ValueError("unused motor must be disabled")
        arrays = {name: np.array([getattr(m, name) for m in msg.motor_cmd[:self.count]], dtype=np.float32)
                  for name in ("q", "dq", "tau", "kp", "kd")}
        return MotorPacket(PolicyCommand(arrays["q"], arrays["kp"], arrays["kd"]), arrays["dq"], arrays["tau"])

    def decode_state(self, msg):
        if not isinstance(msg, self.state_type) or msg.crc != self.crc.Crc(msg):
            raise ValueError("state CRC/type")
        if self.robot == "h1" and (list(msg.head) != [0xFE, 0xEF] or msg.level_flag != 0xFF):
            raise ValueError("state header")
        if self.robot == "g1" and msg.mode_pr != 0:
            raise ValueError("state PR mode")
        q = np.array([m.q for m in msg.motor_state[:self.count]], dtype=np.float32)
        dq = np.array([m.dq for m in msg.motor_state[:self.count]], dtype=np.float32)
        quat = np.asarray(msg.imu_state.quaternion, dtype=np.float64)
        omega = np.asarray(msg.imu_state.gyroscope, dtype=np.float64)
        if not np.isfinite(np.r_[q, dq, quat, omega]).all() or abs(np.linalg.norm(quat)-1) > .05:
            raise ValueError("invalid state/IMU")
        if self.robot == "h1":
            # H1's IMU is on the torso. Match the preserved hardware adapter:
            # remove waist yaw and waist angular velocity to recover pelvis data.
            waist = Rotation.from_euler("z", float(q[6])).as_matrix()
            pelvis = quat_to_matrix(quat) @ waist.T
            quat = Rotation.from_matrix(pelvis).as_quat()[[3, 0, 1, 2]]
            omega = waist @ omega - np.array([0., 0., dq[6]])
        state = RobotState(float(msg.tick)/1000., q, dq,
                           {"quat": quat.astype(np.float32), "omega": omega.astype(np.float32),
                            "tick": int(msg.tick), "mode_machine": int(getattr(msg, "mode_machine", 0))})
        state.validate(self.count)
        return state


class Feedback:
    """A fresh receipt cannot disguise a publisher replaying a frozen tick."""
    def __init__(self, timeout=.1):
        self.timeout = timeout
        self.state = None
        self.received = self.advanced = 0.
        self.lock = threading.Lock()

    def update(self, state, now):
        with self.lock:
            old = None if self.state is None else self.state.extra["tick"]
            tick = state.extra["tick"]
            if old is not None and tick != old and (tick-old) % (2**32) >= 2**31:
                raise ValueError("feedback tick moved backwards")
            if old is None or tick != old:
                self.advanced = now
            self.state, self.received = state, now

    def read(self, now):
        with self.lock:
            if self.state is None:
                raise RuntimeError("No DDS feedback")
            age, tick_age = now-self.received, now-self.advanced
            if age > self.timeout or tick_age > self.timeout:
                raise RuntimeError(f"Stale DDS feedback: receipt={age:.4f}s tick={tick_age:.4f}s")
            state = self.state
            return RobotState(state.timestamp, state.q.copy(), state.dq.copy(),
                              dict(state.extra, feedback_age=age, tick_age=tick_age, received_at=self.received))


def connect_loopback(domain):
    if not 1 <= domain <= 232:
        raise ValueError("Use an isolated DDS domain from 1 to 232")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(domain, "lo")


class DdsBackend:
    def __init__(self, robot, domain, timeout=.1):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        self.wire = Wire(robot)
        self.feedback = Feedback(timeout)
        self.bad_states = self.state_count = self.command_count = 0
        connect_loopback(domain)
        self.publisher = ChannelPublisher("rt/lowcmd", self.wire.command_type)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", self.wire.state_type)
        self.subscriber.Init(self.on_state, 1)

    def on_state(self, msg):
        try:
            state = self.wire.decode_state(msg)
            self.feedback.update(state, time.monotonic())
            self.state_count += 1
        except (ValueError, TypeError):
            self.bad_states += 1

    def wait_state(self, timeout=15.):
        deadline = time.monotonic()+timeout
        while self.feedback.state is None and time.monotonic() < deadline:
            time.sleep(.01)
        return self.read_state()

    def read_state(self):
        return self.feedback.read(time.monotonic())

    def write_command(self, command, state):
        latest = self.read_state()
        if time.monotonic()-state.extra["received_at"] > self.feedback.timeout:
            raise RuntimeError("Policy command was computed from stale DDS feedback")
        msg = self.wire.encode_command(command, latest.extra["mode_machine"])
        if not self.publisher.Write(msg):
            raise RuntimeError("DDS command publication failed")
        self.command_count += 1

    def close(self):
        self.subscriber.Close()
        self.publisher.Close()
