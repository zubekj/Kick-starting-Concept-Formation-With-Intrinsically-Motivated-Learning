import matplotlib.pyplot as plt
import numpy as np


# Load actuator maps and weight data
actuator_maps = np.load("www/StoredGripActuatorMap.npy", allow_pickle=True)
weight_data = np.load("storage/000999/weights.npy", allow_pickle=True)[0]

# Define constants
num_touch_sensors = 40
num_proprio_sensors = 5
num_eyepos_coords = 2
grid_side = 5
grid_size = grid_side**2

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(10, 4))


def plot_touch_map(actuator_maps, num_touch_sensors, grid_size, grid_side, ax):
    """Plot the touch map on the given axis."""
    touch_map = actuator_maps[:num_touch_sensors, :grid_size]
    touch_map = touch_map.reshape(4, 10, grid_side, grid_side)
    touch_map = touch_map.transpose(2, 1, 3, 0).reshape(
        grid_side * 10, grid_side * 4
    )
    ax.imshow(touch_map, aspect=2 / 5)
    # yticks = np.linspace(4.5, 44.5, 5)
    # xticks = np.linspace(2, 17, 5)
    # ax.set_yticks(yticks, np.arange(5))
    # ax.set_xticks(xticks, np.arange(5))
    for x in np.linspace(-0.5, 19.5, 6)[1:-1]:
        ax.plot([x, x], [-0.5, 49.5], c="white")
    for x in np.linspace(-0.5, 19.5, 21)[1:-1]:
        ax.plot([x, x], [0, 49.5], lw=0.2,c="white")
    for y in np.linspace(-0.5, 49.5, 6)[1:-1]:
        ax.plot([-0.5, 19.5], [y, y], c="white")
    for y in np.linspace(-0.5, 49.5, 51)[1:-1]:
        ax.plot([-0.5, 19.5], [y, y], lw=0.2,c="white")
    # ax.set_xlim(0, 19)
    # ax.set_ylim(0, 49)


def plot_posture_maps(
    actuator_maps,
    num_touch_sensors,
    num_proprio_sensors,
    grid_size,
    grid_side,
    ax,
):
    """Plot the posture maps on the given axis."""
    ax.set_aspect("equal")
    posture_map = actuator_maps[
        num_touch_sensors : (num_touch_sensors + num_proprio_sensors),
        grid_size : (2 * grid_size),
    ]
    segment_lengths = [0.3, 0.3, 0.3, 0.2, 0.2]
    postures = []
    for k in range(grid_size):
        right_points, left_points = [[0, 0]], [[0, 0]]
        angles = posture_map[:, k]
        angle = 0.5 * np.pi
        for i, (length, angle_offset) in enumerate(
            zip(segment_lengths, angles)
        ):
            angle += angle_offset
            right_point = length * np.array([np.cos(angle), np.sin(angle)])
            if i > 2:
                angle = np.pi * 0.5 - angle
            left_point = length * np.array([np.cos(angle), np.sin(angle)])
            right_points.append(right_point + right_points[i])
            left_points.append(left_point + left_points[i])
        postures.append([np.vstack(right_points), np.vstack(left_points)])
    postures = np.stack(postures)
    postures = (postures - postures.min()) / np.ptp(postures)
    for k in range(grid_size):
        i, j = k % grid_side, k // grid_side
        right_points, left_points = postures[k] + [i, grid_side - j]
        ax.plot(*right_points.T, c="black")
        ax.scatter(*right_points.T, s=2, c="black")
        ax.plot(*left_points.T, c="black")
        ax.scatter(*left_points.T, s=2, c="black")
    ax.set_xticks(np.arange(grid_side) + 0.3, np.arange(grid_side))
    ax.set_yticks(
        np.arange(1, grid_side + 1) + 0.5, np.arange(grid_side)[::-1]
    )
    ax.set_xlim(-0.2, grid_side - 0.2)
    ax.set_ylim(1, grid_side + 1)


def plot_eyepos_maps(
    actuator_maps,
    num_touch_sensors,
    num_proprio_sensors,
    num_eyepos_coords,
    grid_size,
    grid_side,
    ax,
):
    """Plot the eyepos  maps on the given axis."""
    ax.set_aspect("equal")
    eyepos_map = actuator_maps[
        -num_eyepos_coords:,
        (2 * grid_size) :,
    ]

    eyepos_map = eyepos_map.reshape(num_eyepos_coords, grid_side, grid_side)
    eyepos_map = eyepos_map.transpose(0, 2, 1)

    ax.scatter(*eyepos_map)


def plot_weights(
    weights,
    grid_size,
    grid_side,
    prototype,
    joint,
    ax,
):
    policies = weights["policy"]
    _, out_size = policies.shape
    policies = policies.reshape(grid_size * 3, 5, -1).transpose(2, 1, 0)
    policy = (
        policies[prototype, joint, :]
        .reshape(grid_side, grid_side, 3)
        .transpose(2, 0, 1)
    )
    policy = np.hstack([x for x in policy])
    ax.imshow(policy[::-1, :])


# Plot touch and posture maps
plot_touch_map(
    actuator_maps,
    num_touch_sensors,
    grid_size,
    grid_side,
    axes[0],
)
plot_posture_maps(
    actuator_maps,
    num_touch_sensors,
    num_proprio_sensors,
    grid_size,
    grid_side,
    axes[1],
)
plot_eyepos_maps(
    actuator_maps,
    num_touch_sensors,
    num_proprio_sensors,
    num_eyepos_coords,
    grid_size,
    grid_side,
    axes[2],
)


# for joint in range(5):
#     fig, axes = plt.subplots(11, 11, figsize=(8, 3))
# 
#     for i, ax in enumerate(axes.flatten()):
#         ax.set_axis_off()
#         row, col = i // 11, i % 11
#         p = (row - 1) * 10 + (col - 1)
#         if col > 0 and row > 0:
#             plot_weights(
#                 weight_data,
#                 grid_size,
#                 grid_side,
#                 prototype=p,
#                 joint=joint,
#                 ax=ax,
#             )
#         else:
#             if col == 0 and row != 0:
#                 ax.text(0.5, 0.5, f"{row}")
#             elif row == 0 and col != 0:
#                 ax.text(0.5, 0.5, f"{col}")
#     fig.suptitle(f"Joint n.{joint}")
#     fig.text(0.1, 0.1, "1")
#     fig.tight_layout(pad=0.1)

# Display the plots
plt.show()
