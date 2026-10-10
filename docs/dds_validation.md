# DDS validation (v0.2.0)

Four real-time, separate-process DDS episodes completed locomotion → imitation → locomotion and at least six seconds of control after handback. All used actual SDK2 messages on `lo`; simulator and controller process IDs differ in each report.

| Episode | Duration | Post-settling max tilt | Forward command / measured mean | Feedback-age p99 |
| --- | ---: | ---: | ---: | ---: |
| [H1 standing](validation/dds/h1_standing/metrics.json) | 13.02 s | 2.39° | 0.00 / 0.005 m/s | 4.94 ms |
| [H1 walking](validation/dds/h1_walking/metrics.json) | 16.00 s | 3.16° | 0.60 / 0.372 m/s | 4.89 ms |
| [G1 standing](validation/dds/g1_standing/metrics.json) | 31.14 s | 1.27° | 0.00 / 0.001 m/s | 5.40 ms |
| [G1 walking](validation/dds/g1_walking/metrics.json) | 34.14 s | 2.36° | 0.30 / 0.251 m/s | 5.30 ms |

Post-settling metrics exclude the first second after handback. `standing` means active zero-command locomotion; balance corrections and stepping remain possible. H1 retains appreciable forward-speed tracking error. The documented walking acceptance tolerance is `max(0.15 m/s, 50% of the command)`.

All four runs recorded zero invalid commands/states, matching sent/received command counts, no command watchdog event, and real-time factors within 0.00001 of 1.0. These four demonstrations are not a randomized success-rate benchmark.

Feedback age measures receipt-to-command-publication time, not one-way network latency. Command ages in the trajectory and `max_command_age_ms` are sampled at the 50 Hz recording rate; they are not an exhaustive latency bound. The command watchdog checks at every physics step. Full timing and acceptance thresholds appear in each metrics file and manifest.

## Checks

- 22 local package tests passed, including actual GO/HG IDL serialization, CRC rejection, motor modes, unused slots, H1 IMU frame conversion, frozen/backward tick detection, switching, and H1 adapter equivalence.
- 25 preserved H1 operator regression tests passed. These use synthetic feedback and do not launch DDS.
- CI has a separate SDK2/CycloneDDS contract-test job. It does not launch the full policy demos or have access to the local H1 checkpoints.
- Deliberate command/feedback interruptions are recorded separately as expected failures. See [fault-check evidence](validation/dds/fault_checks.json).

## Reproduce and inspect

Use the [DDS guide](dds.md). The release archive contains each process’s NPZ trajectory and log, the merged rendering trajectory, manifests, and metrics. Each manifest hashes the exact policy/transport sources used; assets have independent checksums. Videos identify `MuJoCo + DDS` in the overlay.

The setup uses preset initial poses, flat ground, no external supports, and a disabled H1 obstacle. It tests asynchronous loopback DDS with measured process scheduling. Physical startup, robot Ethernet, actuator modeling accuracy, arbitrary gait-phase entry and hardware safety remain outside these results.
