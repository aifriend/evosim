"""evosim: training general-purpose embodied agents with Gymnasium and MuJoCo."""

from gymnasium.envs.registration import register

register(
    id="GripperCube-v0",
    entry_point="evosim.envs.gripper_cube:GripperCubeEnv",
    max_episode_steps=200,
)

from evosim.envs.gripper_cube import GripperCubeEnv  # noqa: E402,F401
