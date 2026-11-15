"""Smoke tests for the custom environments."""

import gymnasium as gym
import numpy as np

import evosim  # noqa: F401  (registers custom environments)


def test_gripper_cube_registered():
    env = gym.make("GripperCube-v0")
    assert env.spec.id == "GripperCube-v0"
    env.close()


def test_gripper_cube_step_contract():
    env = gym.make("GripperCube-v0")
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert "cube_z" in info
        if terminated or truncated:
            obs, info = env.reset()
    env.close()


def test_gripper_cube_deterministic_reset():
    env = gym.make("GripperCube-v0")
    obs1, _ = env.reset(seed=7)
    obs2, _ = env.reset(seed=7)
    assert np.allclose(obs1, obs2)
    env.close()


def test_gripper_cube_xy_varies_across_resets():
    # The cube owns the first free joint, so qpos[0:2] is its x/y position.
    env = gym.make("GripperCube-v0")
    base = env.unwrapped
    xy = []
    for seed in range(5):
        env.reset(seed=seed)
        xy.append(base.data.qpos[0:2].copy())
    env.close()
    xy = np.array(xy)
    assert np.all(xy.std(axis=0) > 0), "cube x/y should be randomized on reset"
    assert np.all(np.abs(xy) <= 0.02 + 1e-9)
