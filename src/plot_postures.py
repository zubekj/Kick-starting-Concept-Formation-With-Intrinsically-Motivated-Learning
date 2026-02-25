#!/usr/bin/env python
"""Visualization module for trajectory animation with weight maps.

This module provides functionality for creating animated visualizations of
robot arm trajectories, including somatosensory, proprioceptive, and visual
weight map representations. It supports both online rendering and saving
animations to GIF files.

Example:
    Run from command line::

        python SMAnimation.py -e 1 -r 0 -g 5 --online

Attributes:
    matplotlib: Configured to use 'qtagg' backend for rendering.
"""

import argparse
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.interpolate import splev, splprep

from params import Parameters
from SMGraphs import GraphManager
from SMMain import build_episode_dataset


warnings.filterwarnings("error")

matplotlib.use("qtagg")


def generate_offset_points(points, distance=0.05):
    """Generate points offset by perpendicular normals.

    Computes offset points along perpendicular normals to the path defined
    by input points. Useful for creating parallel curves.

    Args:
        points: Array-like of shape (N, 2) containing 2D point coordinates.
        distance: Offset distance from original points. Positive values
            offset to the left of the path direction. Defaults to 0.05.

    Returns:
        np.ndarray: Array of shape (N, 2) containing offset point
            coordinates. Returns copy of input if N=1, empty array if N=0.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 2:
        return points.copy() if n == 1 else np.empty((0, 2), dtype=float)

    p_prev = np.empty_like(points)
    p_next = np.empty_like(points)
    p_prev[0], p_prev[1:] = points[0], points[:-1]
    p_next[-1], p_next[:-1] = points[-1], points[1:]

    d = p_next - p_prev
    lengths = np.hypot(d[:, 0], d[:, 1])
    nonzero = lengths > 0

    normals = np.zeros_like(d)
    normals[nonzero, 0] = -d[nonzero, 1] / lengths[nonzero]
    normals[nonzero, 1] = d[nonzero, 0] / lengths[nonzero]

    return points + normals * distance


def interp(points, n=10):
    """Interpolate points using B-spline.

    Fits a B-spline curve through the given points and samples it at
    uniformly spaced parameter values.

    Args:
        points: Array of shape (M, 2) containing control points.
        n: Number of interpolated points to generate. Defaults to 10.

    Returns:
        np.ndarray: Array of shape (n, 2) containing interpolated points.
    """
    tck, _ = splprep(points.T, s=0)
    return np.vstack(splev(np.linspace(0, 1, n), tck)).T


def plot_polyline(angles, lengths):
    """Compute arm and gripper coordinates from joint angles.

    Performs forward kinematics to compute the positions of arm segments
    and gripper fingers based on joint angles and segment lengths.

    Args:
        angles: Sequence of joint angles in degrees. Last two angles
            correspond to gripper fingers.
        lengths: Sequence of segment lengths corresponding to each joint.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Arm coordinates of shape (N-2, 2).
            - np.ndarray: Gripper coordinates of shape (5, 2).
    """
    angles = np.array(angles)
    angles[0] += 90
    angles[-2:] *= [-1, 1]

    def compute_coords(ang):
        angle_sum = np.cumsum(np.radians(ang))
        x, y = np.zeros(len(ang) + 1), np.zeros(len(ang) + 1)
        x[1:] = lengths * np.cos(angle_sum)
        y[1:] = lengths * np.sin(angle_sum)
        return np.vstack((np.cumsum(x), np.cumsum(y))).T

    arm_coords = compute_coords(angles)
    angles[-2:] *= -1
    x2y2 = compute_coords(angles)[-3:]
    grip_coords = np.vstack([arm_coords[-2:][::-1], x2y2])

    return arm_coords[:-2], grip_coords


def get_sensors_coords(grip_coords, n=20):
    """Get sensor coordinates around gripper.

    Generates sensor positions along both sides of the gripper by
    interpolating offset curves.

    Args:
        grip_coords: Array of shape (M, 2) containing gripper coordinates.
        n: Number of sensor points per side. Defaults to 20.

    Returns:
        np.ndarray: Array of shape (2*n, 2) containing sensor positions
            arranged in a continuous loop around the gripper.
    """
    n_2 = n // 2
    pts1 = interp(generate_offset_points(grip_coords, -0.1), n)[::-1]
    pts2 = interp(generate_offset_points(grip_coords, 0.1), n)
    return np.vstack([pts1[n_2:], pts2, pts1[:n_2]])


def plot_somatosensory(axes, weights, px, py, sensor_points):
    """Plot somatosensory weight visualization.

    Displays sensor activations as scatter points with sizes proportional
    to weight values.

    Args:
        axes: Dictionary mapping axis names to matplotlib Axes objects.
        weights: Dictionary containing weight arrays with 'ssensory' key.
        px: X-coordinate index in the weight map.
        py: Y-coordinate index in the weight map.
        sensor_points: Array of shape (N, 2) containing sensor positions.
    """
    sensors = weights["ssensory"][px, py]
    ax = axes["ssensory"]
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.5)

    idcs = np.where(sensors > 1e-5)
    ax.scatter(*sensor_points[idcs].T, c="black", s=0.5 + 60 * sensors[idcs])
    ax.scatter(*sensor_points.T, c="black", s=0.1)
    ax.set_axis_off()


def plot_proprioceptive(axes, weights, px, py, g):
    """Plot proprioceptive weight visualization.

    Displays gripper configuration based on proprioceptive weight values.

    Args:
        axes: Dictionary mapping axis names to matplotlib Axes objects.
        weights: Dictionary containing weight arrays with 'proprio' key.
        px: X-coordinate index in the weight map.
        py: Y-coordinate index in the weight map.
        g: GraphManager instance for generating gripper coordinates.
    """
    angles = weights["proprio"][px, py]
    grip = g.generate_gripper(angles)
    ax = axes["proprio"]
    ax.clear()
    ax.set_xlim(-0.2, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.scatter(*grip.T, c="black")
    ax.plot(*grip.T, c="black")
    ax.set_axis_off()


def plot_retina(axes, weights, px, py):
    """Plot visual/retina weight visualization.

    Displays normalized visual weight map as an image.

    Args:
        axes: Dictionary mapping axis names to matplotlib Axes objects.
        weights: Dictionary containing weight arrays with 'visual' key.
        px: X-coordinate index in the weight map.
        py: Y-coordinate index in the weight map.
    """
    retina = weights["visual"][px, py]
    retina = (retina - retina.min()) / np.ptp(retina)
    ax = axes["visual"]
    ax.clear()
    ax.imshow(retina, aspect="auto")
    ax.set_axis_off()


def load_and_process_data(trajectory_file="trajectories.csv", weight_file="weights.npy"):
    """Load trajectory and weight data, reshape weights appropriately.

    Loads data from files and reshapes weight arrays to expected
    dimensions for visualization.

    Args:
        trajectory_file: Path to CSV file containing trajectory data.
            Defaults to "trajectories.csv".
        weight_file: Path to NPY file containing weight arrays.
            Defaults to "weights.npy".

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Trajectory data.
            - bool: True if sensor columns ('s0') exist in data.
            - dict: Reshaped weight arrays with keys 'proprio',
                'ssensory', and 'visual'.
    """
    df = pd.read_csv(trajectory_file)
    has_sensors = "s0" in df.columns
    weights = np.load(weight_file, allow_pickle=True)[0]

    dim = weights["proprio"].shape[0]
    weights["proprio"] = weights["proprio"].reshape(dim, 10, 10)
    weights["proprio"] = weights["proprio"].transpose(1, 2, 0)[:, :, -2:]

    dim = weights["ssensory"].shape[0]
    weights["ssensory"] = weights["ssensory"].reshape(dim, 10, 10)
    weights["ssensory"] = weights["ssensory"].transpose(1, 2, 0)

    weights["visual"] = weights["visual"].reshape(10, 10, 3, 10, 10)
    weights["visual"] = weights["visual"].transpose(3, 4, 0, 1, 2)

    return df, has_sensors, weights


def create_figure_layout(g):
    """Create figure with subplot grid layout.

    Sets up the complete figure layout with all required subplots for
    trajectory visualization, weight maps, and sensor displays.

    Args:
        g: GraphManager instance for generating sensor points.

    Returns:
        tuple: A tuple containing:
            - plt.Figure: The created figure.
            - dict: Dictionary mapping axis names to Axes objects.
            - np.ndarray: X-axis limits as [min, max].
            - np.ndarray: Y-axis limits as [min, max].
            - np.ndarray: Sensor point coordinates of shape (40, 2).
    """
    xlims, ylims = np.array([-2, 4]), np.array([-3, 3])
    gridsize = (4, 3)
    ratio = gridsize[0] / gridsize[1]
    dim = 8

    fig = plt.figure(figsize=(dim, dim * ratio))

    axes = {}
    for name, pos in [("pmap", [3, 0]), ("smap", [3, 1]), ("vmap", [3, 2])]:
        axes[name] = plt.subplot2grid(gridsize, pos, 1, 1, fig=fig)
        axes[name].set_axis_off()

    titles = {
        "proprio": "Proprioception",
        "ssensory": "Somatosensory",
        "visual": "Foveal vision",
    }
    for i, (name, title) in enumerate(titles.items()):
        axes[name] = plt.subplot2grid(gridsize, [2, i], 1, 1, fig=fig)
        axes[name].set_title(title)

    sensor_points = g.generate_sensor_points(40)

    axes["video"] = plt.subplot2grid(gridsize, [0, 1], 2, 2, fig=fig, aspect="equal")
    axes["label"] = plt.subplot2grid(gridsize, [0, 0], 1, 1, fig=fig, aspect="equal")
    axes["label"].set_axis_off()

    axes["traces"] = plt.subplot2grid(gridsize, [1, 0], 1, 1, fig=fig, aspect="equal")
    axes["traces"].set_xlim(-0.5, 9.5)
    axes["traces"].set_ylim(-0.5, 9.5)
    axes["traces"].set_xticks(np.arange(10), [])
    axes["traces"].set_yticks(np.arange(10), [])
    axes["traces"].grid()

    fig.tight_layout(pad=0.1)
    return fig, axes, xlims, ylims, sensor_points


def setup_ax(axes, key, xlims=None, ylims=None):
    """Clear and configure axis limits.

    Args:
        axes: Dictionary mapping axis names to matplotlib Axes objects.
        key: Key identifying the axis to configure.
        xlims: Optional tuple or array of (min, max) for x-axis limits.
        ylims: Optional tuple or array of (min, max) for y-axis limits.
    """
    ax = axes[key]
    ax.clear()
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_axis_off()


def add_marker(ax, point, width, height):
    """Add rectangular marker to axis.

    Args:
        ax: Matplotlib Axes object to add marker to.
        point: Array-like of (x, y) coordinates for marker center.
        width: Width of the rectangular marker.
        height: Height of the rectangular marker.
    """
    rp = Rectangle(point - 0.5, width=width, height=height, fc="#fff0", ec="red")
    ax.add_patch(rp)


def update_maps(g, wfile, font_size, axes, px, py):
    """Update all weight map visualizations.

    Refreshes proprioceptive, visual, and somatosensory map displays
    with current prototype position markers.

    Args:
        g: GraphManager instance for generating map visualizations.
        wfile: Path to weight file for map generation.
        font_size: Font size for labels (unused in current implementation).
        axes: Dictionary mapping axis names to matplotlib Axes objects.
        px: X-coordinate index for marker position.
        py: Y-coordinate index for marker position.
    """
    setup_ax(axes, "pmap")
    g.proprio_map(ax=axes["pmap"], wfile=wfile)
    add_marker(axes["pmap"], np.array([py, px]) + 0.5, 1, 1)

    setup_ax(axes, "vmap")
    g.visual_map(ax=axes["vmap"], wfile=wfile)
    add_marker(axes["vmap"], np.array([py, 10 - px - 1]) * 10 + 0.5, 10, 10)

    setup_ax(axes, "smap")
    g.somatosensory_map(ax=axes["smap"], wfile=wfile)
    add_marker(axes["smap"], np.array([py, px]) + 0.5, 1, 1)


class TrajectoryAnimator:
    """Handles trajectory animation with visual feedback.

    Creates animated visualizations of robot arm trajectories including
    arm movement, sensor activations, and representation traces on
    weight maps.

    Attributes:
        COLORS: Class-level color definitions for visualization elements.
        axes: Dictionary of matplotlib Axes objects.
        fig: Matplotlib Figure object.
        has_sensors: Whether sensor data is available.
        xlims: X-axis limits for video display.
        ylims: Y-axis limits for video display.
        font_size: Font size for text labels.
        episode_id: Current episode identifier.
        rep: Current repetition number.
        fast: Whether to use fast rendering mode.
        params: Parameters instance.
        lines1: List of arm segment line artists.
        lines2: List of gripper segment line artists.
        scatters: List of sensor scatter artists.
        reps: Dictionary of representation scatter artists.
        traces: Dictionary of trace line artists.
        anim: Current FuncAnimation instance.
        episode_df: DataFrame containing episode data.
        conditions_df: DataFrame containing condition data.
    """

    COLORS = {"goal": "#cc4", "touch": "#c44", "proprio": "#44c"}

    def __init__(
        self,
        params,
        font_size,
        axes,
        fig,
        has_sensors,
        xlims,
        ylims,
        episode_id,
        rep,
        trajectories,
        fast=True,
    ):
        """Initialize the trajectory animator.

        Args:
            params: Parameters instance containing simulation settings.
            font_size: Font size for text labels.
            axes: Dictionary mapping axis names to matplotlib Axes.
            fig: Matplotlib Figure object for animation.
            has_sensors: Whether sensor data columns exist.
            xlims: X-axis limits as (min, max) for video display.
            ylims: Y-axis limits as (min, max) for video display.
            episode_id: Unique identifier for the episode.
            rep: Repetition number within the episode.
            fast: If True, use simplified trace rendering. Defaults to True.
        """
        self.axes = axes
        self.fig = fig
        self.has_sensors = has_sensors
        self.xlims = xlims
        self.ylims = ylims
        self.font_size = font_size
        self.episode_id = episode_id
        self.rep = rep
        self.fast = fast
        self.params = params

        self.lines1, self.lines2, self.scatters = [], [], []
        self.reps, self.traces = {}, {}
        self._initialized = False
        self.anim = None
        self.trajectories = trajectories

        self.episode_df, self.conditions_df = build_episode_dataset(params)

    def _init_artists(self, n, episode_id, trajectory):
        """Initialize plot artists for animation.

        Args:
            n: Number of trajectory frames.
            episode_id: Episode identifier for loading GIF.
            trajectory: DataFrame containing trajectory data.
        """
        video_ax = self.axes["video"]
        traces_ax = self.axes["traces"]
        label_ax = self.axes["label"]

        for i in range(n):
            (line1,) = video_ax.plot([], [], c="black", marker="o", zorder=-100 + n)
            (line2,) = video_ax.plot([], [], c="black", marker="o", zorder=-100 + n)
            self.lines1.append(line1)
            self.lines2.append(line2)

            if self.has_sensors:
                scatter = video_ax.scatter([], [], c="red", zorder=-100 + n - 1)
                self.scatters.append(scatter)

        self._init_traces(traces_ax, n)
        self._init_reps(traces_ax, n)
        traces_ax.legend(loc="center left", bbox_to_anchor=(1, 0.7), title="Reps")

        self._init_labels(label_ax, trajectory, episode_id)
        self._initialized = True

    def _init_traces(self, ax, n):
        """Initialize trace lines.

        Args:
            ax: Matplotlib Axes object for traces.
            n: Number of trajectory frames.
        """
        colors = {
            "g": self.COLORS["goal"],
            "ss": self.COLORS["touch"],
            "p": self.COLORS["proprio"],
        }
        if self.fast:
            self.traces = {
                k: ax.plot([999], [999], lw=0.5, c=c)[0] for k, c in colors.items()
            }
        else:
            self.traces = {
                k: [ax.plot([999], [999], c=c)[0] for _ in range(n - 1)]
                for k, c in colors.items()
            }

    def _init_reps(self, ax, n):
        """Initialize representation scatter plots.

        Args:
            ax: Matplotlib Axes object for representations.
            n: Number of trajectory frames (unused).
        """
        configs = [
            ("g", "h", self.COLORS["goal"], "goal"),
            ("ss", "*", self.COLORS["touch"], "somatosen"),
            ("p", "*", self.COLORS["proprio"], "proprio"),
        ]
        for key, marker, color, label in configs:
            self.reps[key] = ax.scatter(
                999, 999, marker=marker, fc=color, ec="#000", lw=0.5, s=300, label=label
            )

    def _init_labels(self, ax, trajectory, episode_id):
        """Initialize episode labels and image.

        Args:
            ax: Matplotlib Axes object for labels.
            trajectory: DataFrame containing trajectory data.
            episode_id: Episode identifier for loading GIF.
        """
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        tsl = trajectory.ets.iloc[-1]
        t_tot_frames = self.trajectories.shape[0]

        episode_file = f"episode_{episode_id}_{self.rep + 1}.gif"
        framel_file = f"episode_{episode_id}_{self.rep + 1}.png"
        with Image.open(episode_file) as gif:

            tot_frames = gif.n_frames
            start = self.params.drop_first_n_steps + self.params.policy_selection_steps
            print(tsl)
            print(t_tot_frames)
            print(tot_frames)
            print(start)
            gif.seek(
                +start + tsl,
            )
            self.framel = np.array(gif)[150:250, 50:150]
            self.episode_template = ax.imshow(
                self.framel, extent=[0, 6, 4, 10], zorder=900
            )
            plt.imsave(framel_file, self.framel)

        query = self.episode_df.query(f"index=={episode_id}")
        objs = ["blue cube", "red triangle", "green cube"]
        obj = objs[query.context.iloc[0] - 1]
        stretch = query.stretch.iloc[0]
        rot = np.degrees(query.rotation.iloc[0]).round(0)

        self.episode_label = ax.text(
            0,
            0,
            f" Object: {obj}\nStretch: {stretch}\nrotation: {rot}°\n",
            fontdict={"size": self.font_size},
            zorder=800,
            verticalalignment="bottom",
        )

        self.axes["proprio"].set_title("Proprioception")
        self.axes["ssensory"].set_title("Somatosensory")
        self.axes["visual"].set_title("Foveal vision")
        self.axes["proprio"].text(
            -1.2,
            0.7,
            "Current prototypes",
            fontdict={"size": self.font_size},
            rotation=90,
        )
        self.axes["pmap"].text(
            -2.5,
            1.5,
            "Representation grids",
            fontdict={"size": self.font_size},
            rotation=90,
        )

    def clear(self):
        """Clear all animation artists."""
        for line in self.lines1 + self.lines2:
            line.set_data([], [])
        for scatter in self.scatters:
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_sizes([])

    def animate(self, trajectory):
        """Create and return animation for trajectory.

        Args:
            trajectory: DataFrame containing trajectory data with columns
                for timestamps, joint angles, sensor values, and
                representation coordinates.

        Returns:
            FuncAnimation: Animation object that can be displayed or saved.
        """
        n = trajectory.shape[0]
        ts_vals = trajectory.ts.to_numpy()
        data = trajectory.iloc[:, 1:6].to_numpy()
        sensor_data = trajectory.iloc[:, 6:46].to_numpy() if self.has_sensors else None

        ss_data = trajectory.loc[:, ["touch_x", "touch_y"]].to_numpy()[:, ::-1]
        p_data = trajectory.loc[:, ["proprio_x", "proprio_y"]].to_numpy()[:, ::-1]
        g_data = trajectory.loc[:, ["prototype_x", "prototype_y"]].to_numpy()[:, ::-1]

        ss_pp = p_pp = None
        if not self.fast:
            ss_pp = np.stack([ss_data[:-1], ss_data[1:]], axis=1)
            p_pp = np.stack([p_data[:-1], p_data[1:]], axis=1)

        if not self._initialized or len(self.lines1) != n:
            self._clear_artists()
            self._init_artists(n, self.episode_id, trajectory)

        indices = ts_vals.astype(int)
        all_angles = np.degrees(data[indices])
        polylines = [plot_polyline(ang, [1, 1, 1, 0.5, 0.5]) for ang in all_angles]
        alphas = 0.01 + 0.99 * np.exp(-np.linspace(-50, 0, n) ** 2)

        offsets_pts, sizes_arr = [], []
        if self.has_sensors:
            all_sensors = sensor_data[indices]
            for i, (_, grip_coords) in enumerate(polylines):
                offsets_pts.append(get_sensors_coords(grip_coords))
                sizes_arr.append(100 * all_sensors[i])

        def update(frame_idx):
            print(f"frame: {frame_idx:4d}")
            artists = []
            for i in range(frame_idx + 1):
                arm_coords, grip_coords = polylines[i]
                alpha = alphas[i]
                loc_alphas = 0.01 + 0.99 * np.exp(-np.linspace(-15, 0, i + 1) ** 2)

                self.lines1[i].set_data(arm_coords[:, 0], arm_coords[:, 1])
                self.lines1[i].set_alpha(alpha)
                self.lines2[i].set_data(grip_coords[:, 0], grip_coords[:, 1])
                self.lines2[i].set_alpha(alpha)

                self._update_traces(i, ss_data, p_data, ss_pp, p_pp, loc_alphas)
                self._update_reps(i, ss_data, p_data, g_data, loc_alphas, n)

                artists.extend([self.lines1[i], self.lines2[i], *self.reps.values()])
                artists.extend(self._get_trace_artists())

                if self.has_sensors:
                    self.scatters[i].set_offsets(offsets_pts[i])
                    self.scatters[i].set_sizes(sizes_arr[i])
                    self.scatters[i].set_alpha(0.2 + 0.8 * alpha)
                    artists.append(self.scatters[i])

                artists.extend([self.episode_template, self.episode_label])

            return artists

        self.anim = FuncAnimation(
            self.fig, update, frames=n, interval=50, blit=True, repeat=False
        )
        return self.anim

    def _clear_artists(self):
        """Remove all existing artists."""
        for line in self.lines1 + self.lines2:
            line.remove()
        for scatter in self.scatters:
            scatter.remove()
        self.lines1.clear()
        self.lines2.clear()
        self.scatters.clear()
        self.traces.clear()
        self.reps.clear()

    def _update_traces(self, i, ss_data, p_data, ss_pp, p_pp, loc_alphas):
        """Update trace line data.

        Args:
            i: Current frame index.
            ss_data: Somatosensory coordinate data.
            p_data: Proprioceptive coordinate data.
            ss_pp: Somatosensory point pairs for non-fast mode.
            p_pp: Proprioceptive point pairs for non-fast mode.
            loc_alphas: Alpha values for trace transparency.
        """
        if self.fast:
            self.traces["ss"].set_data(*ss_data[: i + 1].T)
            self.traces["p"].set_data(*p_data[: i + 1].T)
        else:
            for k, tr in enumerate(self.traces["ss"][: i + 1]):
                tr.set_data(*ss_pp[k].T)
                tr.set_linewidth(0.2 + 4 * loc_alphas[k])
                tr.set_alpha(loc_alphas[k])
            for k, tr in enumerate(self.traces["p"][: i + 1]):
                tr.set_data(*p_pp[k].T)
                tr.set_linewidth(0.2 + 4 * loc_alphas[k])
                tr.set_alpha(loc_alphas[k])

    def _update_reps(self, i, ss_data, p_data, g_data, loc_alphas, n):
        """Update representation scatter plots.

        Args:
            i: Current frame index.
            ss_data: Somatosensory coordinate data.
            p_data: Proprioceptive coordinate data.
            g_data: Goal coordinate data.
            loc_alphas: Alpha values for marker transparency.
            n: Total number of frames.
        """
        for key, data, base_size in [("ss", ss_data, 50), ("p", p_data, 0)]:
            self.reps[key].set_offsets(data[: i + 1])
            sizes = base_size + (250 if key == "ss" else 300) * loc_alphas[: i + 1]
            self.reps[key].set_sizes(sizes)

            fc = np.tile(self.reps[key].get_facecolor(), [n, 1])
            ec = np.tile(self.reps[key].get_edgecolor(), [n, 1])
            fc[: i + 1, 3] = loc_alphas[: i + 1]
            ec[: i + 1, 3] = loc_alphas[: i + 1]
            self.reps[key].set_facecolor(fc[: i + 1])
            self.reps[key].set_edgecolor(ec[: i + 1])

        self.reps["g"].set_offsets([g_data[i]])

    def _get_trace_artists(self):
        """Get all trace artists as flat list.

        Returns:
            list: List of trace line artists.
        """
        if self.fast:
            return list(self.traces.values())
        return [seg for trace in self.traces.values() for seg in trace]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process episode and goal identifiers.")
    parser.add_argument(
        "-e",
        "--episode_id",
        type=int,
        help="Unique identifier for the episode",
    )
    parser.add_argument(
        "-r",
        "--rep",
        type=int,
        help="Number of episode repetition",
    )
    parser.add_argument(
        "-g",
        "--goal_id",
        type=int,
        help="Unique identifier for the goal",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="Only list the table of episodes and goals",
    )
    parser.add_argument(
        "-o",
        "--online",
        action="store_true",
        help="Render online",
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Fast rendering of representations",
    )
    args = parser.parse_args()

    params = Parameters()
    g = GraphManager(None, params)
    font_size = 11

    df, has_sensors, weights = load_and_process_data(trajectory_file="trajectory_df.csv")
    seeds = df.e_seed.unique()
    wfile = "weights.npy"

    gdf = (
        df.groupby(["e_seed", "episode_id", "goal_id"])
        .first()
        .reset_index()[["e_seed", "episode_id", "goal_id"]]
    )

    if args.list:
        gdf = gdf.groupby(["e_seed", "episode_id"]).count().reset_index()
        gdf = gdf.pivot(index="episode_id", columns="e_seed").reset_index()
        gdf.columns = [
            "_".join(map(str, col)).strip("_") if isinstance(col, tuple) else col
            for col in gdf.columns
        ]
        gdf["type"] = [["B", "R", "G"][i] for i in np.arange(18) % 3]
        print(gdf)
        sys.exit()

    curr_episode_gdf = gdf.query(
        f"episode_id == {args.episode_id} and e_seed == {int(seeds[args.rep])}"
    )

    curr_episode_gdf = curr_episode_gdf.copy()
    curr_episode_gdf.loc[:, "curr_goal"] = ""

    mask = curr_episode_gdf["goal_id"] == args.goal_id
    curr_episode_gdf.loc[mask, "curr_goal"] = "*"

    print()
    print(curr_episode_gdf)
    print()

    if args.goal_id not in curr_episode_gdf["goal_id"].values:
        raise ValueError("The current episode do not reach the this goal number")

    df.loc[:, "ets"] = df.groupby(["e_seed", "episode_id"]).cumcount()
    mask = df.e_seed == seeds[args.rep]
    mask &= df.episode_id == args.episode_id
    trajectories = df.loc[mask].copy()
    mask &= df.goal_id == args.goal_id
    trajectory = df.loc[mask].copy()

    fig, axes, xlims, ylims, sensor_points = create_figure_layout(g)
    setup_ax(axes, "video", xlims, ylims)
    px, py = int(trajectory.prototype_x.iat[0]), int(trajectory.prototype_y.iat[0])

    update_maps(g, wfile, font_size, axes, px, py)
    plot_proprioceptive(axes, weights, px, py, g)
    plot_somatosensory(axes, weights, px, py, sensor_points)
    plot_retina(axes, weights, px, py)

    animator = TrajectoryAnimator(
        params,
        font_size,
        axes,
        fig,
        has_sensors,
        xlims,
        ylims,
        args.episode_id,
        args.rep,
        fast=args.fast,
        trajectories=trajectories,
    )
    anim = animator.animate(trajectory)

    name = f"e{args.episode_id:02d}_g{args.goal_id:02d}_r{args.rep}_{px}{py}"
    if args.online:
        plt.show()
    else:
        writer = matplotlib.animation.PillowWriter(fps=2, metadata=["loop", "1"])
        anim.save(
            filename=f"{name}.gif",
            writer=writer,
        )
        animator.fig.savefig(f"{name}.png", dpi=300)
