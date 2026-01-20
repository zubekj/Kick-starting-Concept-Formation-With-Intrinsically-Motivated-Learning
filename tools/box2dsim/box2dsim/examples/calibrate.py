#####
import box2dsim
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# Import box2dsim to ensure the environment is registered
_ = box2dsim

# Initialize random number generator
rng = np.random.RandomState(62)

# Create the Box2DSim environment with specific random object parameters
env = gym.make(
    "Box2DSimOneArmOneEye-v0",
    rand_obj_params={
        "stretch_conditions": [1, 1.5, 2],
        "rotation_conditions": np.pi * np.array([0, 0.25, 0.5]),
        "pos": [2, 0.4],
    },
).unwrapped

# Iterate over different world configurations
for world in range(1, 4):
    env.set_world(world)  # Set the current world
    stime = 100  # Simulation time steps
    init_action = np.pi * np.array([0.0, 0.0, 0.0, 0.5, 0])  # Initial action
    action = np.pi * np.array([0.0, 0.0, 0.0, 0, 0])  # Action to be taken

    env.reset()  # Reset the environment
    for t in range(stime):
        if t > stime * 0.6:  # Apply actions after 60% of the time steps
            action += 0.0 * rng.randn(5)  # Add randomness to the action
            action[:3] = np.pi * np.array([0.2, -0.4, -0.4])  # Set arm actions
            action[3:] = np.pi * np.array([0.2, 0.2])  # Set eye actions

            print(action)  # Print the current action

            env.step(action)  # Take a step in the environment
            env.render()  # Render the environment
            plt.pause(0.1)  # Pause for a short duration
#####
