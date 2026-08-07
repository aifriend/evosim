#!/usr/bin/env python3
"""Train an agent on a Gymnasium/MuJoCo task with Stable Baselines3.

Examples:
    python scripts/train.py --env HalfCheetah-v5 --timesteps 1000000
    python scripts/train.py --env GripperCube-v0 --algo ppo --n-envs 8
"""

import argparse
from pathlib import Path

import evosim  # noqa: F401  (registers custom environments)
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

ALGOS = {"ppo": PPO, "sac": SAC}

ENV_IDS = [
    "CartPole-v1",
    "GripperCube-v0",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Ant-v5",
    "Humanoid-v5",
    "Reacher-v5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="HalfCheetah-v5", choices=ENV_IDS)
    parser.add_argument("--algo", default="ppo", choices=list(ALGOS))
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None,
                        help="defaults to <algo>_<env>_<seed>")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.algo == "sac":
        action_space = gym.make(args.env).action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise SystemExit(
                f"SAC requires a continuous action space; "
                f"{args.env} has {type(action_space).__name__}. Use --algo ppo."
            )
    run_name = args.run_name or f"{args.algo}_{args.env}_{args.seed}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(args.env, n_envs=args.n_envs, seed=args.seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    eval_env = VecNormalize(
        make_vec_env(args.env, n_envs=1, seed=args.seed + 1),
        norm_obs=True, norm_reward=False, training=False,
    )

    model = ALGOS[args.algo](
        "MlpPolicy",
        train_env,
        seed=args.seed,
        tensorboard_log=str(run_dir / "tb"),
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=args.algo,
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "eval"),
            eval_freq=max(25_000 // args.n_envs, 1),
            n_eval_episodes=10,
        ),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(run_dir / "final_model")
    train_env.save(str(run_dir / "vecnormalize.pkl"))
    print(f"Done. Artifacts in {run_dir}")


if __name__ == "__main__":
    main()
