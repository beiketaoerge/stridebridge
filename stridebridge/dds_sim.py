"""Real-time MuJoCo process. Only received DDS packets produce motor torques."""
import json
import os
import threading
import time

import mujoco
import numpy as np

from .dds import Wire, connect_loopback
from .geometry import tilt
from .models import build_simulation


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(path)


class DdsSimulator:
    def __init__(self, args):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        self.args, self.sim = args, build_simulation(args)
        self.wire = Wire(args.robot)
        if set(self.sim.ids) != set(self.wire.enabled):
            raise ValueError("MuJoCo actuators do not match the DDS motor contract")
        self.lock = threading.Lock()
        self.packet = None
        self.received = 0.
        self.command_count = self.bad_commands = self.state_count = 0
        self.watchdog_at = None
        self.sensor_slices = {}
        for name in ("imu_quat", "imu_gyro", "imu_acc"):
            i = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if i < 0:
                raise ValueError("Missing IMU sensor: " + name)
            start = self.sim.model.sensor_adr[i]
            self.sensor_slices[name] = slice(start, start+self.sim.model.sensor_dim[i])
        connect_loopback(args.domain_id)
        self.publisher = ChannelPublisher("rt/lowstate", self.wire.state_type)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowcmd", self.wire.command_type)
        self.subscriber.Init(self.on_command, 1)

    def on_command(self, msg):
        try:
            packet = self.wire.decode_command(msg)
            with self.lock:
                self.packet, self.received = packet, time.monotonic()
                self.command_count += 1
        except (ValueError, TypeError):
            self.bad_commands += 1

    def publish_state(self):
        sim = self.sim
        msg = self.wire.new_state()
        if self.args.robot == "h1":
            msg.head[:] = [0xFE, 0xEF]
            msg.level_flag = 0xFF
        for j, i in enumerate(sim.ids):
            motor = msg.motor_state[i]
            motor.mode = self.wire.motor_mode(int(i))
            motor.q = float(sim.data.qpos[sim.qidx[j]])
            motor.dq = float(sim.data.qvel[sim.vidx[j]])
            motor.ddq = float(sim.data.qacc[sim.vidx[j]])
            motor.tau_est = float(sim.data.actuator_force[sim.aidx[j]])
        for field, name in (("quaternion", "imu_quat"), ("gyroscope", "imu_gyro"), ("accelerometer", "imu_acc")):
            getattr(msg.imu_state, field)[:] = sim.data.sensordata[self.sensor_slices[name]].tolist()
        msg.tick = int(round(sim.data.time*1000)) % (2**32)
        msg.crc = self.wire.crc.Crc(msg)
        if not self.publisher.Write(msg):
            raise RuntimeError("DDS state publication failed")
        self.state_count += 1

    def apply(self, packet, age):
        sim = self.sim
        q, dq = sim.data.qpos[sim.qidx], sim.data.qvel[sim.vidx]
        if age > self.args.packet_timeout:
            torque = -8.*dq
            if self.watchdog_at is None:
                self.watchdog_at = float(sim.data.time)
        else:
            ids, c = sim.ids, packet.command
            torque = c.kp[ids]*(c.q[ids]-q) + c.kd[ids]*(packet.dq[ids]-dq) + packet.tau[ids]
        bounds = sim.model.actuator_ctrlrange[sim.aidx]
        limited = sim.model.actuator_ctrllimited[sim.aidx].astype(bool)
        torque[limited] = np.clip(torque[limited], bounds[limited, 0], bounds[limited, 1])
        sim.data.ctrl[:] = 0
        sim.data.ctrl[sim.aidx] = torque
        mujoco.mj_step(sim.model, sim.data)

    def run(self):
        sim, args = self.sim, self.args
        rows, lateness = [], []
        started = None
        deadline = time.monotonic()+args.max_seconds+30
        next_tick = time.monotonic()
        step, failure = 0, None
        try:
            while time.monotonic() < deadline:
                if (args.output/"controller_result.json").exists():
                    break
                now = time.monotonic()
                with self.lock:
                    packet, received, count = self.packet, self.received, self.command_count
                mujoco.mj_forward(sim.model, sim.data)
                if not (args.fault == "state-loss" and 3 <= sim.data.time < 3.3):
                    self.publish_state()
                if packet is None:
                    # Initial scene setup only; after the first command physics
                    # advances independently of policy inference and message arrival.
                    time.sleep(.002)
                    next_tick = time.monotonic()
                    continue
                if started is None:
                    started = now
                    next_tick = now
                age = now-received
                lateness.append(max(0., now-next_tick))
                if step % 10 == 0:
                    state = sim.read()
                    rows.append({"time": state.timestamp, "wall_time": now-started,
                                 "qpos": sim.data.qpos.copy(), "qvel": sim.data.qvel.copy(),
                                 "velocity": state.extra["velocity"], "tilt": tilt(state.extra["quat"]),
                                 "height": state.extra["height"], "command_age": age,
                                 "command_count": count, "target": packet.command.q.copy()})
                self.apply(packet, age)
                step += 1
                if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
                    failure = "nonfinite_physics"
                    break
                if self.watchdog_at is not None and sim.data.time >= self.watchdog_at+.06:
                    failure = "command_timeout"
                    break
                if tilt(sim.data.qpos[3:7]) >= 45 or sim.data.qpos[2] < (.45 if args.robot == "g1" else .65):
                    failure = "fall_threshold"
                    break
                if sim.data.time >= args.max_seconds:
                    failure = "episode_timeout"
                    break
                next_tick += sim.model.opt.timestep
                delay = next_tick-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            else:
                failure = "wall_timeout"
        except Exception as exc:
            failure = type(exc).__name__ + ": " + str(exc)
        finally:
            end_wall = time.monotonic()
            if rows:
                np.savez_compressed(args.output/"simulator.npz", **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})
            result = {"pid": os.getpid(), "failure": failure, "physics_steps": step,
                      "simulation_seconds": float(sim.data.time),
                      "wall_seconds": end_wall-started if started else 0.,
                      "commands_received": self.command_count, "invalid_commands": self.bad_commands,
                      "states_published": self.state_count, "command_watchdog_at_s": self.watchdog_at,
                      "physics_lateness_p99_ms": float(np.percentile(lateness, 99)*1000) if lateness else None}
            save_json(args.output/"simulator_result.json", result)
            print(json.dumps(result), flush=True)
            self.subscriber.Close()
            self.publisher.Close()
        return failure is None
