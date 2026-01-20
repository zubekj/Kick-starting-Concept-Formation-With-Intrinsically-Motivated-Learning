#####
import box2dsim
import gymnasium as gym
import numpy as np
from scipy import interpolate

_ = box2dsim

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
    env.render()
    action = actions_interp[t]
    o, *_ = env.step(action)
