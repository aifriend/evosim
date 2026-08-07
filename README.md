# evosim

Training general-purpose tasks and embodied (bodily) agents with modern
reinforcement learning, built on [Gymnasium](https://gymnasium.farama.org/)
and [MuJoCo](https://mujoco.org/).

The repo showcases two sides of embodied AI:

- **Locomotion (bodily agents)** — learning to control multi-joint simulated
  bodies (`HalfCheetah`, `Hopper`, `Walker2d`, `Ant`, `Humanoid`).
- **Manipulation (bodily objects)** — learning to act on objects in the
  world, including a custom MuJoCo environment (`GripperCube-v0`) where a
  two-finger gripper must grasp and lift a cube.

## Features

- Custom MuJoCo environment defined from scratch (XML model + Gymnasium API)
- One training entry point for both built-in and custom tasks
- PPO and SAC via [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- Vectorized training, periodic evaluation, checkpoints, TensorBoard logging
- Evaluation script with video recording

## Repository layout

```
evosim/                   # Python package
  envs/
    assets/gripper_cube.xml   # MuJoCo model for the custom task
    gripper_cube.py           # Gymnasium env: grasp & lift a cube
scripts/
  train.py                # train PPO/SAC on any supported task
  evaluate.py             # evaluate a saved policy, optionally record video
tests/
  test_envs.py            # smoke tests for the custom environment
```

## Installation

Requires Python 3.10+ and ffmpeg (only for video recording).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
```

## Quickstart

Train a locomotion agent:

```bash
python scripts/train.py --env HalfCheetah-v5 --algo ppo --timesteps 1000000
```

Train the custom manipulation task:

```bash
python scripts/train.py --env GripperCube-v0 --algo ppo --timesteps 500000 --n-envs 8
```

Monitor training:

```bash
tensorboard --logdir runs
```

Evaluate and record a video:

```bash
python scripts/evaluate.py \
    --env GripperCube-v0 \
    --model runs/ppo_GripperCube-v0_0/best/best_model.zip \
    --episodes 5 --record
```

## Environments

| ID                | Kind         | Description                                   |
|-------------------|--------------|-----------------------------------------------|
| `GripperCube-v0`  | manipulation | Custom: gripper grasps and lifts a cube       |
| `Reacher-v5`      | manipulation | 2-DOF arm reaches a random target             |
| `HalfCheetah-v5`  | locomotion   | 2D planar runner                              |
| `Hopper-v5`       | locomotion   | 2D one-legged hopping                         |
| `Walker2d-v5`     | locomotion   | 2D bipedal walking                            |
| `Ant-v5`          | locomotion   | 3D quadruped locomotion                       |
| `Humanoid-v5`     | locomotion   | 3D bipedal humanoid (hard exploration)        |

### GripperCube-v0 (custom)

- **Observation**: joint positions/velocities plus the cube position relative
  to the gripper.
- **Action**: 3 continuous values — lift height, left/right finger closure —
  mapped onto MuJoCo position actuators.
- **Reward**: shaped on lifted cube height, plus a success bonus when the
  cube is held above the target height.
- **Randomization**: the cube's resting position is jittered on every reset.

## Results

_TODO: add learning curves and videos of trained agents here._

## Roadmap

- [ ] Learning curves + agent videos for each environment
- [ ] More manipulation tasks (push, reach-grasp, in-hand rotation)
- [ ] Domain randomization experiments
- [ ] Multi-task / generalist policy training across environments

## License

MIT — see [LICENSE](LICENSE).
