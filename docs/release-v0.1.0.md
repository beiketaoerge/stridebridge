# StrideBridge v0.1.0

Continuous switching between pretrained locomotion and motion-imitation policies
for Unitree H1 and G1, with active balance before and after the motion.

- Preserves the latest H1 controller, coordinated assist and operator regressions.
- Adds G1 29-DoF velocity/dance adapters using downloadable Unitree policies.
- Provides four continuous MuJoCo videos: standing → motion → standing and
  walking → pause → motion → walking for both robots.
- Includes reference-clock ownership, smooth target/gain blending and measured
  post-transition balance/velocity results.

All four final flat-ground demo episodes passed the documented criteria. The
11 package tests and 25 retained H1 operator tests passed locally. These are
deterministic demonstration trials, not a robustness or hardware benchmark.

The current switching manager is deterministic. A learned transition policy
remains the next research objective. Public G1 assets are downloadable with
`scripts/fetch_assets.py`; the H1 demo requires the user's existing local bundles.
No local H1 weights or motion data are redistributed in these release assets.

Each MP4 contains the full trajectory at 25 fps and real-time playback speed.
The ZIP contains all four recorded state trajectories, metrics and runtime
manifests. `SHA256SUMS` verifies the downloadable files. H1 videos use the
step-over reference on flat ground with the board removed.
