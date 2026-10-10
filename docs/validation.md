# Release validation

This page records the v0.1.0 direct simulation baseline. See
[DDS validation](dds_validation.md) for the v0.2.0 separate-process DDS results.

Recorded on 2026-10-10. These are four deterministic demonstration episodes,
each with one entry and one exit. They are not a randomized success-rate study.

| Episode | Duration | Maximum tilt after settling | Minimum base height over run | Forward command / measured mean after settling |
| --- | --- | --- | --- | --- |
| [H1 standing](validation/h1_standing/metrics.json) | 13.00 s | 2.230° | 0.909 m | 0 / −0.001 m/s |
| [H1 walking](validation/h1_walking/metrics.json) | 16.00 s | 3.231° | 0.910 m | 0.600 / 0.359 m/s |
| [G1 standing](validation/g1_standing/metrics.json) | 31.12 s | 1.326° | 0.723 m | 0 / 0.004 m/s |
| [G1 walking](validation/g1_walking/metrics.json) | 34.12 s | 3.038° | 0.723 m | 0.300 / 0.238 m/s |

Every final episode completed its clip and handback and continued for six seconds.
The first second after handback is a settling interval; post-settling metrics use
the following five seconds. G1 follows the upstream dance interval 2.8–24.0 s.
H1 consumes the supplied 154-frame, 50 Hz reference. Its coordinated assist remains
enabled. Both walking scenarios use zero commanded velocity before entry and
restore the forward command after return. During the zero-command intervals,
locomotion inference and motor-target updates continue at 50 Hz; the H1 gait
phase also continues. The robot remains free to step or correct its balance.
The `standing` scenario uses this same active zero-command behavior throughout
its locomotion phases.

The demo's explicit pass criteria are: handback completes within the trial timeout;
the robot stays below 45° tilt and above 0.45 m (G1) / 0.65 m (H1) base height;
six seconds of post-handoff control are recorded; post-settling tilt stays below
15°; standing mean planar speed is below 0.15 m/s, or walking mean forward-speed
error is at most `max(0.15 m/s, 50% of the command)`. The walking tolerance is broad:
the H1 trial demonstrates locomotion continuity with visible tracking error.
Thresholds and full-precision measurements are in each JSON.

No elastic band, root constraint, support wrench, per-frame pose reset or video
cut is used in these simulations. Rendering replays logged physics states; it
does not generate a new physical trajectory. GIF previews are the full walking
videos resampled to 8 fps, retaining real-time duration. MP4 files are 960×640,
25 fps. Trajectories, metrics and source/environment manifests accompany the release.

## Environment and tests

Inference: Python 3.8.20, NumPy 1.24.4, MuJoCo 3.2.3, SciPy 1.10.1,
ONNX Runtime 1.19.2 (G1), PyTorch 2.4.1+cu121 on CPU (H1). Both inference engines
use a single compute thread. See each episode's `manifest.json` for source hashes,
asset-manifest hash, arguments and versions. Asset files are verified before use.

The 11 package tests cover stable gating, monotonic handoff, command feedback,
reset, nonfinite feedback, negative gains, invalid timing, G1 executed-action
history, owned motion timing, and H1 adapter equivalence against the preserved
controller. The separate 25 H1 operator regressions also passed after path
portability changes; their [report](validation/h1_operator_portable.json) records
the controller hash. The equivalence test uses synthetic feedback and actual H1
weights; it does not establish dynamics equivalence of different simulators.

## Development failures retained

The [development ledger](validation/development_trials.json) retains the initial
experiments separately from the final four episodes:

- The first G1 trial used RL Gym's older 29-DoF robot XML and fell before entry.
  Unitree MuJoCo's matching deployment model resolved that configuration mismatch.
- Two subsequent G1 trials executed the dance but fell during handback. Resetting
  the velocity history alone did not solve the issue. Feeding executed targets
  back in that policy's action coordinates resolved it in the tested scenarios.
- The initial H1 0.3 m/s walking trial remained balanced but measured only about
  0.027 m/s after handback. Its original pass flag checked balance only. It would
  fail the release's later velocity criterion. The final H1 walking demonstration
  uses a 0.6 m/s command and reports its measured 0.359 m/s.

The retained H1 obstacle benchmark is described in [H1 lineage](h1.md). The new
flat-ground episodes are not obstacle-crossing evidence. No hardware runs or
perturbation/generalization claims are made for this release.
