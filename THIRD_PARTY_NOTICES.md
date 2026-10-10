# Third-party notices

The switching foundation is derived from [JamesW546/h1switch](https://github.com/JamesW546/h1switch),
commit `66a0da8bf1e4728fb663b8e72e880a8119838260`, copyright 2026 Yiming Wang,
under the MIT license reproduced in [LICENSE](LICENSE). The preserved H1 work
extends that controller with the project's existing adapters and assist logic.

The G1 adapter follows the observation, action, motion loading and torso-anchor
interfaces of Unitree RL Lab, copyright Unitree Robotics Co., Ltd., Apache-2.0.
The Python implementation adds owned motion timing, executed-target history
synchronization and single-process simulation. [Upstream license](licenses/unitree_rl_lab.txt).

Downloaded Unitree RL Gym assets retain its BSD-3-Clause license and inherited
notices. [License](licenses/unitree_rl_gym.txt).
Downloaded Unitree MuJoCo assets retain its BSD-3-Clause license.
[License](licenses/unitree_mujoco.txt).

The downloader retains asset provenance, revisions and individual checksums.
The repository's MIT license does not replace the licenses or terms of external
weights, motion data, robot meshes, or the user's locally imported H1 bundle.
