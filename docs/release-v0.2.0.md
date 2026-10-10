# StrideBridge v0.2.0 — H1 and G1 over DDS

H1 and G1 now run locomotion → motion imitation → locomotion through actual
Unitree SDK2 DDS messages. A controller process publishes `rt/lowcmd`; a separate
MuJoCo process applies those commands and publishes `rt/lowstate`. Physics runs
independently at 500 Hz and both policies update at 50 Hz.

- Four continuous videos cover active balance and walking before/after motion
  for both robots, with DDS feedback/command-age overlays.
- H1 uses GO messages and torso-to-pelvis IMU conversion. G1 uses HG messages,
  serial pitch/roll mode, and the received machine-mode field.
- Zero commanded velocity keeps inference, feedback and H1 gait phase running.
- CRC/mode checks reject invalid packets. Watchdogs reject stale feedback,
  frozen ticks, old inference snapshots and expired commands.
- Separate process logs, traffic counters, source/asset hashes and trajectories
  accompany the videos. Setup and fault-injection commands are documented.

All four final flat-ground DDS episodes passed the documented handoff,
post-transition balance/velocity and real-time criteria. The local suite passed
22 package tests and 25 preserved H1 operator tests. Deliberate H1 command loss
and G1 feedback loss produced the expected watchdog failures.

These recordings use localhost DDS and MuJoCo. They do not establish physical
robot safety, Ethernet behavior or robustness to disturbances. Initial poses are
preset, and H1's obstacle is disabled. There are no external support forces or
mid-episode state resets. Local H1 weights and motion files are not redistributed.

Each MP4 is the full recorded trajectory at real-time playback speed. The ZIP
contains controller/simulator recordings and logs, metrics, manifests and the
fault-check evidence. `SHA256SUMS` verifies the release assets.
