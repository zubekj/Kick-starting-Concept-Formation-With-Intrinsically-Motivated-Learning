import os

import box2dsim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

_ = box2dsim

rng = np.random.RandomState(62)
env = gym.make(
    "Box2DSimOneArmOneEye-v0",
    rand_obj_params={
        "stretch_conditions": [1, 1.5, 2],
        "rotation_conditions": np.pi * np.array([0, 0.25, 0.5]),
        "pos": [2, 0.4],
    },
).unwrapped

# Enable interactive mode for plotting
plt.ion()

# Set random seed for reproducibility
seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
env.set_seed(seed)
stime = 100
trials = 3

# Define random actions
actions = np.pi * env.rng.uniform(-0.5, 0.5, [10, 5])

# Set up the plot
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(121)
screen = ax.imshow(np.zeros([2, 2]), vmin=-0.1, vmax=.2, cmap=plt.cm.binary)
ax.set_axis_off()
ax.set_title("Saliency")
ax1 = fig.add_subplot(122)
fov = ax1.imshow(np.zeros([2, 2, 3]))
ax1.set_axis_off()
ax1.set_title("Fovea")

# Run the simulation
for q in [3, 1, 2]:
    for k in range(trials):
        env.set_world(q)
        env.reset()
        for t in range(stime // 5):
            if t % (stime // 5) == 0:
                action = (
                    0.5
                    * np.pi
                    * env.rng.uniform(0, 1, [5])
                    * [-1, -1, -1, 1, 1]
                )
            env.render()

            env.renderer.ax.set_title(
                "%s object"
                % env.world_object_names[env.world_id][0].capitalize()
            )

            observation, *_ = env.step(action)
            eye = observation["VISUAL_SENSORS"]
            sal = observation["VISUAL_SALIENCY"]

            fov.set_array(eye)
            screen.set_array(sal)
            fig.canvas.draw()
