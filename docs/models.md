# Model catalog

Verified on 2026-10-10 from upstream repository contents. A policy checkpoint,
a reference motion and a robot mesh are different artifacts; an imitation demo
needs all three plus the matching observation/action definitions.

| Project | Relevant artifacts | Use in StrideBridge |
| --- | --- | --- |
| [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/deploy/robots/g1_29dof/config/policy) | G1 29-DoF velocity ONNX, dance_102 and gangnam_style ONNX, reference CSVs and deploy YAMLs | Downloaded and hash checked. Velocity + dance_102 tested. Gangnam weights downloaded but not integrated/tested. |
| [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b/deploy/pre_train) | H1 and G1 TorchScript locomotion policies named `motion.pt` | Downloaded baselines. They are velocity policies, not reference-motion imitators. They are not substituted into the tested H1 pair. |
| [Unitree MuJoCo](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/g1) | G1 29-DoF XML, meshes and scene | The tested G1 physics model. Its motor armature/damping differ from RL Gym's older description. |
| [ASAP](https://github.com/LeCAR-Lab/ASAP#sim2sim) | G1 locomotion and motion-imitation deployment examples and checkpoint paths | Useful alternative with an existing locomotion/imitation switch interface. Not downloaded or adapted in this release. |
| [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) | G1 whole-body tracking training and evaluation pipeline | Suitable for training additional motion experts; the README's evaluation flow expects a trained WandB run. Not an H1 checkpoint substitute. |
| [H2O / OmniH2O](https://github.com/LeCAR-Lab/human2humanoid) | H1 motion tracking / teleoperation code | Relevant H1 research implementation. At commit `750f1fa052641f0fde43669d50cb4e407dabe6c8`, the inspected tree did not contain a ready H1 robot policy in `.pt`, `.pth`, `.jit` or `.onnx` format. No verified drop-in policy is claimed here. Upstream uses CC BY-NC 4.0. |

Public asset inventory and checksums: [assets/manifest.json](../assets/manifest.json).
The downloader fetches three pinned archives (roughly 193 MB compressed total),
then selects about 111 MB of policies, configurations and robot assets. Use
`python scripts/fetch_assets.py --verify-only` to check an existing installation.

The tested H1 pair is the local 0816 DWAQ policy (70 observations × 5 frames,
19 actions in 20 SDK motor slots) and model_30000 (104 observations, 19 actions,
154 reference frames). The coordinated left-leg overlay checks the exact weight
and motion hashes. A new H1 imitation model requires a separate adapter and
transition validation; matching the robot name alone is insufficient.

The official RL Gym H1 baseline has 41 observations and 10 leg actions. Its G1
baseline has 47 observations and 12 leg actions. These contracts differ from
both the local H1 model pair and the G1 29-DoF ONNX pair used in the videos.

Weights, motions and meshes retain upstream terms. The local H1 bundles are
imported from the user's own workspace and stay outside the public Git tree;
this release does not assign them a new redistribution license.
