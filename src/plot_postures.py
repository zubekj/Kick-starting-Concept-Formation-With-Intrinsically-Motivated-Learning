"""
Visualization module for trajectory animation with proprioceptive,
sensory, and visual weight maps.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from scipy.interpolate import splev, splprep

import SMGraphs as g


def generate_offset_points(points, distance=0.05):
    """
    Generate points offset perpendicular to a polyline.

    Args:
        points: Array of 2D points defining the polyline.
        distance: Offset distance (positive = left, negative = right).

    Returns:
        Array of offset points with same shape as input.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return np.empty((0, 2), dtype=float)
    if n == 1:
        return points.copy()

    # Compute previous and next points for tangent estimation
    p_prev = np.empty_like(points)
    p_next = np.empty_like(points)
    p_prev[0] = points[0]
    p_prev[1:] = points[:-1]
    p_next[-1] = points[-1]
    p_next[:-1] = points[1:]

    # Calculate tangent vectors and their lengths
    d = p_next - p_prev
    lengths = np.hypot(d[:, 0], d[:, 1])

    # Compute unit normals (perpendicular to tangents)
    nonzero = lengths > 0
    normals = np.zeros_like(d)
    normals[nonzero, 0] = -d[nonzero, 1] / lengths[nonzero]
    normals[nonzero, 1] = d[nonzero, 0] / lengths[nonzero]

    return points + normals * distance


def interp(points, n=10):
    """
    Interpolate points using B-spline to create smooth curve.

    Args:
        points: Array of 2D control points.
        n: Number of output points.

    Returns:
        Array of n interpolated 2D points.
    """
    tck, u = splprep(points.T, s=0)  # s=0 forces curve through all points
    u_new = np.linspace(0, 1, n)
    return np.vstack(splev(u_new, tck)).T


def plot_polyline(angles, lengths, ss=None):
    """
    Compute polyline coordinates for a multi-segment arm with gripper.

    Args:
        angles: Joint angles in degrees.
        lengths: Segment lengths.
        ss: Unused parameter (reserved for future use).

    Returns:
        Tuple of (arm_points, secondary_points, gripper_points).
    """
    x1, y1 = np.zeros(len(angles)), np.zeros(len(angles))
    angles = np.array(angles)

    # Adjust angles for coordinate system
    angles[1] += 90
    angles[-2:] *= [-1, 1]

    # Compute cumulative angles and segment endpoints
    angle_sum = np.cumsum(np.radians(angles))[1:]
    x1[1:], y1[1:] = lengths * np.cos(angle_sum), lengths * np.sin(angle_sum)
    x1y1 = np.vstack((np.cumsum(x1), np.cumsum(y1))).T

    # Compute mirrored gripper segment
    angles[-2:] *= -1
    angle_sum = np.cumsum(np.radians(angles))[1:]
    x2 = np.zeros(len(angles))
    y2 = np.zeros(len(angles))
    x2[1:], y2[1:] = lengths * np.cos(angle_sum), lengths * np.sin(angle_sum)
    x2y2 = np.vstack((np.cumsum(x2), np.cumsum(y2))).T[-3:]

    # Combine gripper points from both sides
    xy_grip = np.vstack([x1y1[-2:][::-1], x2y2])

    return x1y1, x2y2, xy_grip


def plot_somatosensory(ax, weights, px, py, sensor_points):
    """
    Generate and display the somatosensory activation pattern plot.

    Args:
        ax: Matplotlib axis to plot on.
        weights: Weight dictionary containing 'ssensory' data.
        px: X index into weight grid.
        py: Y index into weight grid.
        sensor_points: Array of sensor point coordinates.
    """
    sensors = weights["ssensory"][px, py]
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.5)
    idcs = np.where(sensors > 1e-5)
    ax.scatter(*sensor_points[idcs].T, c="black", s=0.5 + 60 * sensors[idcs])
    ax.scatter(*sensor_points.T, c="black", s=0.1)
    ax.set_axis_off()


def plot_proprioceptive(ax, weights, px, py):
    """
    Generate and display the proprioceptive gripper pose plot.

    Args:
        ax: Matplotlib axis to plot on.
        weights: Weight dictionary containing 'proprio' data.
        px: X index into weight grid.
        py: Y index into weight grid.
    """
    angles = weights["proprio"][px, py]
    grips = g.generate_gripper(angles)
    ax.clear()
    ax.set_xlim(-1.1, 0.1)
    ax.set_ylim(0.7, 1.3)
    for grip in grips:
        ax.scatter(*grip.T, c="black")
        ax.plot(*grip.T, c="black")
    ax.set_axis_off()


def plot_retina(ax, weights, px, py):
    """
    Generate and display the decoded visual retina image.

    Args:
        ax: Matplotlib axis to plot on.
        weights: Weight dictionary containing 'visual' data.
        px: X index into weight grid.
        py: Y index into weight grid.
    """
    retina = weights["visual"][px, py]
    retina = (retina - retina.min()) / np.ptp(retina)
    ax.clear()
    ax.imshow(retina, aspect="auto")
    ax.set_axis_off()


def load_and_process_data(trajectory_file="trajectories.csv", weight_file="weights.npy"):
    """Load trajectory data and reshape weight matrices."""
    df = pd.read_csv(trajectory_file)
    has_sensors = "s0" in df.columns

    weights = np.load(weight_file, allow_pickle=True)[0]

    dim, _ = weights["proprio"].shape
    weights["proprio"] = weights["proprio"].reshape(dim, 10, 10)
    weights["proprio"] = weights["proprio"].transpose(1, 2, 0)
    weights["proprio"] = weights["proprio"][:, :, -2:]

    dim, _ = weights["ssensory"].shape
    weights["ssensory"] = weights["ssensory"].reshape(dim, 10, 10)
    weights["ssensory"] = weights["ssensory"].transpose(1, 2, 0)

    weights["visual"] = weights["visual"].reshape(10, 10, 3, 10, 10)
    weights["visual"] = weights["visual"].transpose(3, 4, 0, 1, 2)

    return df, has_sensors, weights


def create_figure_layout():
    """Create figure with grid layout and return axes dictionary."""
    xlims = np.array([-0.5, 3])
    ylims = np.array([-2.5, 2])

    gridsize = (3, 5)
    fig = plt.figure(figsize=(8 * 1.666, 8))

    pmap_ax = plt.subplot2grid(gridsize, [0, 0], 1, 1, fig=fig)
    smap_ax = plt.subplot2grid(gridsize, [2, 0], 1, 1, fig=fig)
    vmap_ax = plt.subplot2grid(gridsize, [1, 0], 1, 1, fig=fig)
    pmap_ax.set_axis_off()
    smap_ax.set_axis_off()
    vmap_ax.set_axis_off()

    proprio_ax = plt.subplot2grid(gridsize, [0, 1], 1, 1, fig=fig)
    ssensory_ax = plt.subplot2grid(gridsize, [2, 1], 1, 1, fig=fig)
    visual_ax = plt.subplot2grid(gridsize, [1, 1], 1, 1, fig=fig)

    sensor_points = g.generate_sensor_points(40)
    video_ax = plt.subplot2grid(gridsize, [0, 2], 3, 3, fig=fig, aspect="equal")

    fig.tight_layout(pad=0.3)

    axes = {
        "pmap": pmap_ax,
        "smap": smap_ax,
        "vmap": vmap_ax,
        "proprio": proprio_ax,
        "ssensory": ssensory_ax,
        "visual": visual_ax,
        "video": video_ax,
    }
    return fig, axes, xlims, ylims, sensor_points


def setup_ax(ax, xlims=None, ylims=None):
    ax.clear()
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_axis_off()


def add_marker(ax, point, width, height):
    rp = Rectangle(point - 0.5, width=width, height=height, fc="#fff0", ec="red")
    ax.add_patch(rp)


def update_maps(g, wfile, pmap_ax, vmap_ax, smap_ax, px, py):
    setup_ax(pmap_ax)
    g.proprio_map(ax=pmap_ax, wfile=wfile)
    add_marker(pmap_ax, np.array([py, px]) + [-0.5, 1], 1, 1)

    setup_ax(vmap_ax)
    g.visual_map(ax=vmap_ax, wfile=wfile)
    add_marker(vmap_ax, np.array([py, 10 - px - 1]) * 10 + 0.5, 10, 10)

    setup_ax(smap_ax)
    g.somatosensory_map(ax=smap_ax, wfile=wfile)
    add_marker(smap_ax, np.array([py, px]) + 0.5, 1, 1)


class TrajectoryAnimator:
    def __init__(self, video_ax, fig, has_sensors, xlims, ylims):
        self.video_ax = video_ax
        self.fig = fig
        self.has_sensors = has_sensors
        self.xlims = xlims
        self.ylims = ylims
        self.lines1 = []
        self.lines2 = []
        self.scatters = []
        self._initialized = False
        self._anim = None

    def _init_artists(self, n):
        for _ in range(n):
            (line1,) = self.video_ax.plot([], [], c="black", marker="o")
            (line2,) = self.video_ax.plot([], [], c="black", marker="o")
            self.lines1.append(line1)
            self.lines2.append(line2)
            if self.has_sensors:
                scatter = self.video_ax.scatter([], [], c="red")
                self.scatters.append(scatter)
        self._initialized = True

    def clear(self):
        for line in self.lines1:
            line.set_data([], [])
        for line in self.lines2:
            line.set_data([], [])
        for scatter in self.scatters:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_sizes([])

    def animate(self, trajectory):
        n = trajectory.shape[0]
        ts_vals = trajectory.ts.to_numpy()
        data = trajectory.iloc[:, 1:6].to_numpy()
        sensor_data = trajectory.iloc[:, 6:46].to_numpy() if self.has_sensors else None

        if not self._initialized or len(self.lines1) != n:
            for line in self.lines1:
                line.remove()
            for line in self.lines2:
                line.remove()
            for scatter in self.scatters:
                scatter.remove()
            self.lines1.clear()
            self.lines2.clear()
            self.scatters.clear()
            self._init_artists(n)

        exp_coeff = -((n / 100) ** -2)
        n_minus_1 = n - 1
        indices = ts_vals.astype(int)
        all_angles = np.degrees(data[indices])
        polylines = [plot_polyline(ang, [1, 1, 0.5, 0.5]) for ang in all_angles]
        alphas = 0.01 + 0.99 * np.exp(exp_coeff * (np.arange(n) / n_minus_1 - 1) ** 2)

        offsets_pts = []
        sizes_arr = []
        if self.has_sensors:
            all_sensors = sensor_data[indices]
            for i, (x1y1, x2y2, xy_grip) in enumerate(polylines):
                pts1 = interp(generate_offset_points(xy_grip, distance=-0.1), 20)
                pts2 = interp(generate_offset_points(xy_grip, distance=0.1), 20)
                offsets_pts.append(np.vstack([pts1, pts2]))
                ssensors = all_sensors[i]
                sizes_arr.append(
                    100
                    * np.hstack([ssensors[-10:], ssensors[10:30][::-1], ssensors[:10]])
                )

        def update(frame_idx):
            artists = []
            for i in range(frame_idx + 1):
                x1y1, x2y2, xy_grip = polylines[i]
                alpha = alphas[i]
                self.lines1[i].set_data(x1y1[:, 0], x1y1[:, 1])
                self.lines1[i].set_alpha(alpha)
                self.lines2[i].set_data(x2y2[:, 0], x2y2[:, 1])
                self.lines2[i].set_alpha(alpha)
                artists.extend([self.lines1[i], self.lines2[i]])
                if self.has_sensors:
                    self.scatters[i].set_offsets(offsets_pts[i])
                    self.scatters[i].set_sizes(sizes_arr[i])
                    self.scatters[i].set_alpha(alpha)
                    artists.append(self.scatters[i])
            return artists

        self._anim = FuncAnimation(
            self.fig, update, frames=n, interval=50, blit=True, repeat=False
        )
        plt.show()


df, has_sensors, weights = load_and_process_data()
wfile = "weights.npy"
for tr_id, trajectory in df.groupby(["goal_id", "tr_id"]):
    fig, axes, xlims, ylims, sensor_points = create_figure_layout()
    setup_ax(axes["video"], xlims, ylims)
    py = int(trajectory.prototype_x.iat[0])
    px = int(trajectory.prototype_y.iat[0])

    update_maps(g, wfile, axes["pmap"], axes["vmap"], axes["smap"], px, py)
    plot_proprioceptive(axes["proprio"], weights, px, py)
    plot_somatosensory(axes["ssensory"], weights, px, py, sensor_points)
    plot_retina(axes["visual"], weights, px, py)

    animator = TrajectoryAnimator(axes["video"], fig, has_sensors, xlims, ylims)
    animator.animate(trajectory)
