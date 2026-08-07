#!/usr/bin/env python3
"""CartPole in two lines of training code with Stable Baselines3.

Usage:
    python examples/cartpole_quickstart.py            # train + evaluate
    python examples/cartpole_quickstart.py --render   # also open a live window
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--render", action="store_true",
                    help="render the trained agent in a window")
parser.add_argument("--timesteps", type=int, default=10_000)
args = parser.parse_args()

env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=args.timesteps)

obs = env.reset()[0]
total = 0.0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total += reward
    if terminated or truncated:
        print(f"episode return: {total:.0f}")
        total = 0.0
        obs = env.reset()[0]

env.close()
