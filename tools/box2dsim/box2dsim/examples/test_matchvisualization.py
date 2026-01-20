#####

import box2dsim
import gymnasium as gym
import matplotlib
import numpy as np
from scipy import interpolate


_ = box2dsim

matplotlib.use("agg")

# Set random seed for reproducibility
rng = np.random.RandomState(62)
env = gym.make(
    "Box2DSimOneArmOneEye-v0",
    rand_obj_params={
        "stretch_conditions": [1, 1.5, 2],
        "rotation_conditions": np.pi * np.array([0, 0.25, 0.5]),
        "pos": [2, 0.4],
    },
).unwrapped

# Simulation time and action setup
stime = 120
actions = np.pi * np.array(
    [
        [0.00, 0.00, 0.00, 0.00, 0.00],
        [0.20, -0.30, -0.20, 0.50, 0.00],
        [0.20, -0.30, -0.30, 0.50, 0.00],
        [0.10, -0.30, -0.30, 0.20, 0.30],
        [0.00, -0.30, -0.30, 0.20, 0.50],
        [0.00, -0.30, -0.30, 0.20, 0.50],
        [0.00, -0.30, -0.30, 0.20, 0.50],
    ]
)

# Interpolate actions over the simulation time
actions_interp = np.zeros([stime, 5])

env.reset()
for joint_idx, joint_timeline in enumerate(actions.T):
    x0 = np.linspace(0, 1, len(joint_timeline))
    f = interpolate.interp1d(x0, joint_timeline)
    x = np.linspace(0, 1, stime)
    joint_timeline_interp = f(x)
    actions_interp[:, joint_idx] = joint_timeline_interp

# Run the simulation
env.set_world(3)  # Set the current world
for t in range(stime):
    env.render("offline")
    action = actions_interp[t]
    o, *_ = env.step(action)


match_value = np.random.rand(stime)
cum_match = np.random.rand(stime)
max_match = np.random.rand(stime)

f_vp = np.ones([stime, 2]) * [0, 0]
f_ssp = np.vstack(
    [
        np.ones([stime // 6, 2]) * np.linspace(0, 9, stime // 6).reshape(-1, 1)
        for x in range(6)
    ]
)
f_pp = np.ones([stime, 2]) * [9, 0]
f_ap = np.ones([stime, 2]) * [9, 9]
f_gp = np.ones([stime, 2]) * [0, 9]


# f_gp = np.random.randint(0, 10, (stime, 2))
# f_vp = np.random.randint(0, 10, (stime, 2))
# f_ssp = np.random.randint(0, 10, (stime, 2))
# f_pp = np.random.randint(0, 10, (stime, 2))
# f_ap = np.random.randint(0, 10, (stime, 2))

env.renderer.add_info_to_frames(
    match_value,
    max_match,
    cum_match,
    f_vp,
    f_ssp,
    f_pp,
    f_ap,
    f_gp,
    visual_map_path="visual_map.png",
)
env.renderer.close("demo")
