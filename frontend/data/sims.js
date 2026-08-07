window.SIMS = [
  {
    "id": "CartPole-v1",
    "slug": "cartpole",
    "kind": "classic control",
    "description": "Balance a pole on a moving cart. The classic hello-world of reinforcement learning.",
    "video": "data/videos/cartpole.mp4",
    "policy": "PPO",
    "meanReturn": 500.0,
    "evalEpisodes": 3
  },
  {
    "id": "GripperCube-v0",
    "slug": "grippercube",
    "kind": "manipulation",
    "description": "Custom MuJoCo task: a two-finger gripper must grasp a cube and lift it above a target height.",
    "video": "data/videos/grippercube.mp4",
    "policy": "random (untrained)",
    "meanReturn": -2.3,
    "evalEpisodes": 3
  },
  {
    "id": "Reacher-v5",
    "slug": "reacher",
    "kind": "manipulation",
    "description": "A 2-DOF arm must move its fingertip to a randomly placed target.",
    "video": "data/videos/reacher.mp4",
    "policy": "random (untrained)",
    "meanReturn": -39.7,
    "evalEpisodes": 3
  },
  {
    "id": "HalfCheetah-v5",
    "slug": "halfcheetah",
    "kind": "locomotion",
    "description": "A 2D planar cat-like robot learns to run forward as fast as possible.",
    "video": "data/videos/halfcheetah.mp4",
    "policy": "random (untrained)",
    "meanReturn": -386.2,
    "evalEpisodes": 3
  },
  {
    "id": "Hopper-v5",
    "slug": "hopper",
    "kind": "locomotion",
    "description": "A one-legged 2D robot learns to hop forward without falling.",
    "video": "data/videos/hopper.mp4",
    "policy": "random (untrained)",
    "meanReturn": 36.2,
    "evalEpisodes": 3
  },
  {
    "id": "Walker2d-v5",
    "slug": "walker2d",
    "kind": "locomotion",
    "description": "A 2D biped learns to walk forward with coordinated joints.",
    "video": "data/videos/walker2d.mp4",
    "policy": "random (untrained)",
    "meanReturn": 0.6,
    "evalEpisodes": 3
  },
  {
    "id": "Ant-v5",
    "slug": "ant",
    "kind": "locomotion",
    "description": "A 3D quadruped learns to walk by coordinating four legs.",
    "video": "data/videos/ant.mp4",
    "policy": "random (untrained)",
    "meanReturn": -36.2,
    "evalEpisodes": 3
  },
  {
    "id": "Humanoid-v5",
    "slug": "humanoid",
    "kind": "locomotion",
    "description": "A 3D bipedal humanoid learns to walk \u2014 one of the hardest MuJoCo benchmarks.",
    "video": "data/videos/humanoid.mp4",
    "policy": "random (untrained)",
    "meanReturn": 119.1,
    "evalEpisodes": 3
  }
];
