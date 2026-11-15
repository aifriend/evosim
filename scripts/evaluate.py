#!/usr/bin/env python3
"""Evaluate a trained agent, optionally recording a video.

Examples:
    python scripts/evaluate.py --env HalfCheetah-v5 \
        --model runs/ppo_HalfCheetah-v5_0/final_model.zip
    python scripts/evaluate.py --env GripperCube-v0 \
        --model runs/ppo_GripperCube-v0_0/best/best_model.zip --record
"""

import argparse
from pathlib import Path

import evosim  # noqa: F401  (registers custom environments)
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def algo_name(model_path: Path) -> str:
    """Return "SAC" or "PPO" from the run directory name (<algo>_<env>_<seed>).

    Models are saved as <run>/final_model.zip or <run>/best/best_model.zip, so
    the algorithm is read from whichever parent directory starts with "sac_".
    """
    for parent in model_path.parents:
        if parent.name.lower().startswith("sac_"):
            return "SAC"
    return "PPO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record", action="store_true",
                        help="record mp4 videos into ./videos (requires ffmpeg)")
    parser.add_argument("--vecnormalize", type=Path, default=None,
                        help="path to vecnormalize.pkl saved during training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.record:
        raw = gym.make(args.env, render_mode="rgb_array")
        raw = gym.wrappers.RecordVideo(
            raw, video_folder="videos", episode_trigger=lambda _: True
        )
        env = DummyVecEnv([lambda: raw])
    else:
        env = DummyVecEnv([lambda: gym.make(args.env)])

    vec_path = args.vecnormalize
    if vec_path is None:
        # best/best_model.zip keeps its stats one level up, next to final_model.zip
        for folder in (args.model.parent, args.model.parent.parent):
            candidate = folder / "vecnormalize.pkl"
            if candidate.exists():
                vec_path = candidate
                break
    if vec_path is not None:
        env = VecNormalize.load(str(vec_path), env)
        env.training = False
        env.norm_reward = False

    model_cls = SAC if algo_name(args.model) == "SAC" else PPO
    model = model_cls.load(str(args.model), env=env)

    returns = []
    obs = env.reset()
    for episode in range(args.episodes):
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, _ = env.step(action)
            total += float(reward[0])
            done = bool(dones[0])
        returns.append(total)
        print(f"episode {episode + 1}: return = {total:.2f}")

    mean = sum(returns) / len(returns)
    print(f"mean return over {args.episodes} episodes: {mean:.2f}")
    if args.record:
        print("videos written to ./videos")


if __name__ == "__main__":
    main()
