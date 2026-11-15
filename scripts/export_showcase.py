#!/usr/bin/env python3
"""Export showcase assets for the frontend.

For every supported environment this records a short mp4 (trained policy if a
model exists under runs/, otherwise a random policy labeled as such), measures
the mean return over a few evaluation episodes, and writes a manifest that the
static frontend consumes:

    frontend/data/videos/<slug>.mp4
    frontend/data/sims.js          (window.SIMS = [...])

Usage:
    python scripts/export_showcase.py                     # all environments
    python scripts/export_showcase.py --env HalfCheetah-v5  # just one
"""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import evosim  # noqa: F401  (registers custom environments)
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

OUT_DIR = Path("frontend/data")
VIDEO_DIR = OUT_DIR / "videos"
EVAL_EPISODES = 3
MAX_VIDEO_STEPS = 400

DESCRIPTIONS = {
    "CartPole-v1": "Balance a pole on a moving cart. The classic hello-world of reinforcement learning.",
    "GripperCube-v0": "Custom MuJoCo task: a two-finger gripper must grasp a cube and lift it above a target height.",
    "Reacher-v5": "A 2-DOF arm must move its fingertip to a randomly placed target.",
    "HalfCheetah-v5": "A 2D planar cat-like robot learns to run forward as fast as possible.",
    "Hopper-v5": "A one-legged 2D robot learns to hop forward without falling.",
    "Walker2d-v5": "A 2D biped learns to walk forward with coordinated joints.",
    "Ant-v5": "A 3D quadruped learns to walk by coordinating four legs.",
    "Humanoid-v5": "A 3D bipedal humanoid learns to walk — one of the hardest MuJoCo benchmarks.",
}

KINDS = {
    "CartPole-v1": "classic control",
    "GripperCube-v0": "manipulation",
    "Reacher-v5": "manipulation",
}


def algo_name(model_path: Path) -> str:
    """Return "SAC" or "PPO" from the run directory name (<algo>_<env>_<seed>).

    Models are saved as <run>/final_model.zip or <run>/best/best_model.zip, so
    the algorithm is read from whichever parent directory starts with "sac_".
    """
    for parent in model_path.parents:
        if parent.name.lower().startswith("sac_"):
            return "SAC"
    return "PPO"


def slugify(env_id: str) -> str:
    slug = re.sub(r"-v\d+$", "", env_id).lower()
    return re.sub(r"[^a-z0-9]+", "-", slug).strip("-")


def find_model(env_id: str):
    """Return (model_path, vecnormalize_path) for env_id, or (None, None)."""
    for pattern in (f"runs/*_{env_id}_*/best/best_model.zip",
                    f"runs/*_{env_id}_*/final_model.zip"):
        for model_path in sorted(Path(".").glob(pattern)):
            vec = model_path.parent / "vecnormalize.pkl"
            if not vec.exists():
                vec = model_path.parent.parent / "vecnormalize.pkl"
            return model_path, vec if vec.exists() else None
    return None, None


def record_video(env_id: str, model_path, vec_path, slug: str) -> str:
    """Record up to 3 episodes and keep the longest clip.

    The trained policy sees VecNormalize-normalized observations, matching
    training conditions; otherwise it falls over within a second.
    """
    vn = None
    model = None
    if model_path is not None:
        cls = SAC if algo_name(model_path) == "SAC" else PPO
        model = cls.load(str(model_path))
        if vec_path is not None:
            vn = VecNormalize.load(str(vec_path), DummyVecEnv([lambda: gym.make(env_id)]))
            vn.training = False
            vn.norm_reward = False

    with tempfile.TemporaryDirectory() as tmp:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, video_folder=tmp, name_prefix=slug,
            episode_trigger=lambda _: True,
        )
        episode_lengths = []
        for episode in range(3):
            obs, _ = env.reset(seed=episode)
            for step in range(MAX_VIDEO_STEPS):
                if model is not None:
                    model_obs = vn.normalize_obs(obs[None])[0] if vn else obs
                    action, _ = model.predict(model_obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            episode_lengths.append(step + 1)
            if episode_lengths[-1] >= MAX_VIDEO_STEPS:
                break
        env.close()

        best = episode_lengths.index(max(episode_lengths))
        produced = sorted(Path(tmp).glob(f"{slug}-episode-*.mp4"))
        assert produced, f"no video produced for {env_id}"
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        dest = VIDEO_DIR / f"{slug}.mp4"
        shutil.move(str(produced[min(best, len(produced) - 1)]), dest)
    return f"data/videos/{slug}.mp4"


def mean_return(env_id: str, model_path, vec_path) -> float:
    env = DummyVecEnv([lambda: gym.make(env_id)])
    if vec_path is not None:
        env = VecNormalize.load(str(vec_path), env)
        env.training = False
        env.norm_reward = False
    cls = SAC if model_path and algo_name(model_path) == "SAC" else PPO
    model = cls.load(str(model_path), env=env) if model_path else None

    returns = []
    for episode in range(EVAL_EPISODES):
        obs = env.reset()
        done, total = False, 0.0
        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.array([env.action_space.sample()])
            obs, reward, dones, _ = env.step(action)
            total += float(reward[0])
            done = bool(dones[0])
        returns.append(total)
    env.close()
    return sum(returns) / len(returns)


def load_existing() -> list:
    manifest = OUT_DIR / "sims.js"
    if not manifest.exists():
        return []
    payload = manifest.read_text()
    payload = payload.removeprefix("window.SIMS = ").rstrip().removesuffix(";")
    return json.loads(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", nargs="+", choices=list(DESCRIPTIONS),
                        default=None, help="only export these environments")
    args = parser.parse_args()

    sims = {entry["id"]: entry for entry in load_existing()}
    env_ids = args.env or list(DESCRIPTIONS)
    for env_id in env_ids:
        description = DESCRIPTIONS[env_id]
        slug = slugify(env_id)
        model_path, vec_path = find_model(env_id)
        policy = "random (untrained)"
        if model_path is not None:
            policy = algo_name(model_path)

        print(f"[{env_id}] recording video (policy: {policy}) ...")
        video = record_video(env_id, model_path, vec_path, slug)
        print(f"[{env_id}] evaluating over {EVAL_EPISODES} episodes ...")
        score = mean_return(env_id, model_path, vec_path)

        sims[env_id] = {
            "id": env_id,
            "slug": slug,
            "kind": KINDS.get(env_id, "locomotion"),
            "description": description,
            "video": video,
            "policy": policy,
            "meanReturn": round(score, 1),
            "evalEpisodes": EVAL_EPISODES,
        }
        print(f"[{env_id}] mean return: {score:.1f}")

    # keep a stable ordering: DESCRIPTIONS order first, then any extras
    ordered = [sims.pop(e) for e in DESCRIPTIONS if e in sims] + list(sims.values())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = "window.SIMS = " + json.dumps(ordered, indent=2) + ";\n"
    (OUT_DIR / "sims.js").write_text(payload)
    print(f"\nWrote {OUT_DIR / 'sims.js'} with {len(ordered)} environments.")


if __name__ == "__main__":
    main()
