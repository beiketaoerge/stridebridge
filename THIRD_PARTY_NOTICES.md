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

## DDS dependencies (v0.2.0)

The optional DDS runner imports Unitree SDK2 Python (BSD-3-Clause), tested at
commit `404fe44d76f705c002c97e773276f2a8fefb57e4`, and CycloneDDS Python 0.10.2
(Eclipse Public License 2.0 / Eclipse Distribution License 1.0).
These packages are installed separately; their code and native libraries are not
redistributed in this repository or the recordings archive.
Sources: [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2_python),
[CycloneDDS Python](https://github.com/eclipse-cyclonedds/cyclonedds-python).
