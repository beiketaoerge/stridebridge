# Adapter and transition contract

The common interface is `RobotState(q, dq, extra)` → `PolicyCommand(q, kp, kd)`.
Arrays in a `PolicyCommand` use the simulator/SDK motor order, with positions in
radians and gains in physical units. The policy's input order can differ.
Each adapter converts observations and outputs using its own training contract.

```python
switcher = PolicySwitcher(
    velocity_adapter, motion_adapter,
    initial_policy=velocity_adapter.name,
    transition_seconds=0.4, control_dt=0.02,
    stable_seconds=0.1, max_target_gap=1.0,
    limiter=limiter,
)
switcher.reset(sim.read())
switcher.request_switch(motion_adapter.name)
command, status = switcher.step(sim.read())
sim.step(command)  # exactly one command owner
```

This fragment illustrates one control tick; [demo.py](../stridebridge/demo.py)
contains the complete repeated loop, motion completion and automatic return.

For a new policy, specify robot/joint identity, motor mapping, input ordering,
scales and clipping, reference frame, history/recurrent state, action offsets and
scales, gains and control period. Preserve the previous-action semantics learned
by that policy. Reject an incompatible mapping before inference.

The two public G1 networks use the same 29 joints but different gains, default
poses and action scales. Velocity observations contain five samples per term,
oldest first, with terms concatenated afterward: the input has 480 values.
The mimic input has 154 values. The reference CSV stores root quaternion as
`xyzw`; simulator quaternions are `wxyz`. The mimic anchor is the torso, including
waist yaw/roll/pitch, and its orientation is yaw-aligned at entry.

Motion timing and previous actions are separate concerns. The reference is held
at its selected start during entry blending, then advances at full ownership.
The G1 velocity adapter observes the executed position target converted back
through its own offset/scale. Without that conversion, inactive-policy predictions
can accumulate in the five-frame history and destabilize handback. H1 retains
the independently validated raw-action convention of the latest controller.

H1 defaults are a 0.4 s blend, entry tilt below 15°, leg speed below 2 rad/s,
target gap below 1 rad, and 0.1 s of readiness. Exit allows a 2 rad gap and 0.02 s
readiness, with the source controller's tilt/speed checks. G1 uses a 1.5 rad entry
gap and a tilt gate. These are per-pair baseline settings, not universal stability
conditions. The simulation has a bounded timeout for a gate that never opens.

The walking scenario commands approach motion, stops for 1.5 s before requesting
imitation, then resumes walking one second after handback. It demonstrates
walking before/after a motion; it does not claim arbitrary gait-phase entry.

The switching primitive rejects nonfinite feedback and commands, negative gains
and invalid timing. The original H1 controller additionally owns operator priority,
feedback age/tick validation, CRC handling and stop behavior. The standalone
simulation loop does not open SDK/DDS channels or send hardware commands.
