"""GripperCube: a minimal MuJoCo manipulation task.

A two-finger gripper, actuated along one lifting axis and two closing axes,
must grasp a free cube from a table and lift it above a target height.

Observation: qpos, qvel, and the cube's position relative to the gripper.
Action: 3 continuous values in [-1, 1] mapped onto the actuator ctrlranges
(lift, close left finger, close right finger).
Reward: shaped on cube height, with a bonus for holding the cube above the
success height and a small action-magnitude penalty.
"""

from pathlib import Path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

ASSET_PATH = Path(__file__).parent / "assets" / "gripper_cube.xml"

CUBE_REST_Z = 0.055
SUCCESS_Z = 0.20


class GripperCubeEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 25,
    }

    def __init__(self, frame_skip: int = 20, **kwargs):
        utils.EzPickle.__init__(self, frame_skip=frame_skip, **kwargs)
        MujocoEnv.__init__(
            self,
            model_path=str(ASSET_PATH),
            frame_skip=frame_skip,
            observation_space=None,  # set below, once the model is loaded
            default_camera_config={"distance": 0.8},
            **kwargs,
        )
        obs_dim = self.model.nq + self.model.nv + 3
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        ctrl = self._scale_action(action)
        self.do_simulation(ctrl, self.frame_skip)

        cube_z = self._cube_pos()[2]
        reward = 10.0 * (cube_z - CUBE_REST_Z) - 0.01 * float(np.square(action).sum())

        terminated = bool(cube_z > SUCCESS_Z)
        if terminated:
            reward += 10.0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {"cube_z": cube_z}

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        # Randomize the cube's resting position slightly for robustness.
        qpos[0:2] += self.np_random.uniform(-0.02, 0.02, size=2)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        rel = self._cube_pos() - self._gripper_pos()
        return np.concatenate([self.data.qpos, self.data.qvel, rel]).ravel()

    def _cube_pos(self):
        return self.data.body("cube").xpos

    def _gripper_pos(self):
        return self.data.body("gripper").xpos

    def _scale_action(self, action):
        ctrl_range = self.model.actuator_ctrlrange
        mid = (ctrl_range[:, 0] + ctrl_range[:, 1]) / 2.0
        half = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
        return mid + action * half
