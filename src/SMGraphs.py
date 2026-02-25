from pathlib import Path
from shutil import copyfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from shapely import LineString, MultiLineString
from sklearn.decomposition import PCA

from params import Parameters
from storage import StorageManager


params = Parameters()

c = [0, 0.5, 1]
colors = np.vstack([x.ravel() for x in np.meshgrid(c, c, c)]).T
colors = colors[:-1]
c = np.reshape(colors[1:], (5, 5, 3))
c = np.transpose(c, (1, 0, 2))
c = np.reshape(c, (25, 3))
colors[1:, :] = c
full_palette = LinearSegmentedColormap.from_list("basic", colors)

palette = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="custom", colors=[[0, 1, 0], [1, 0, 0], [0, 0, 1]]
)
internal_side = int(np.sqrt(params.internal_size))
visual_side = int(np.sqrt(params.visual_size / 3))


class GraphManager:
    def __init__(self, sm, params):
        self.sm = sm
        self.params = params

    def store_maps(self):
        map_files = [
            "visual_map.png",
            "comp_map.png",
            "ssensory_map.png",
            "proprio_map.png",
            "goal_frequency_map",
            "log.png",
        ]
        for filename in map_files:
            try:
                copyfile(self.sm.site_dir / filename, self.sm.epoch_dir / filename)
            except (IOError, OSError) as e:
                print(f"Failed to copy {filename}: {e}")

    def update_weight_data(self, weights=None, tag=None, epoch=None):
        if weights is None:
            tag_suffix = f"-{tag}" if tag else ""
            storage_dir = Path(f"storage{tag_suffix}")
            if not storage_dir.is_dir():
                raise Exception(f"{storage_dir} does not exist!")
            if epoch is None:
                epochs = sorted(storage_dir.glob("*"))
                epoch_dir = epochs[-1]
            else:
                epoch_dir = storage_dir / f"{epoch:06d}"
            weights = np.load(epoch_dir / "weights.npy", allow_pickle=True)[0]
        np.save(self.sm.site_dir / "weights", weights)

    def _normalize_array(self, arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    def generate_sensor_points(self, n_sensors):
        a = 0.707
        opoints = np.array(
            [
                [0, 0],
                [-a, 1 - a],
                [-a, 1 - a],
                [-1, 1],
                [-1, 1],
                [-a, 1 - a],
                [-a, 1 - a],
                [0, 0],
            ]
        )
        opoints[4:] *= [[0.9, 1]]
        opoints[4:] += [[0, 0.2]]
        opoints = np.vstack([opoints, opoints[::-1] * [-1, 1]])
        opoints = opoints.reshape(8, 2, 2)
        line = MultiLineString([LineString(opoints[i]) for i in range(8)])
        points = np.array(
            [[i.x, i.y] for i in line.interpolate(np.linspace(0, line.length, n_sensors))]
        )
        points[:, 0] = self._normalize_array(points[:, 0])
        points[:, 1] = self._normalize_array(points[:, 1])
        points = (points - points.min()) / np.ptp(points)
        points *= 0.8
        points += 0.1
        return points

    def generate_sensor_grid(self):

        side = int(np.sqrt(self.params.internal_size))
        n_sensors = self.params.somatosensory_size
        grid = []
        for i in range(side):
            for j in range(side):
                points = self.generate_sensor_points(n_sensors)
                grid.append(points + [j, i])
        grid = np.array(grid)
        grid.reshape(side * side, n_sensors, 2)
        return grid

    def generate_gripper(self, angles):

        init_angles = [0, 0, 0]
        init_angles.extend(angles)
        angles = init_angles
        lengths = [1, 1, 0.5, 0.5]

        angles = np.array(angles)

        angles[-2:] *= [-1, 1]

        angle_sum = np.cumsum(angles)[1:]
        x1, y1 = np.zeros(len(angles)), np.zeros(len(angles))
        x1[1:], y1[1:] = lengths * np.cos(angle_sum), lengths * np.sin(angle_sum)
        arm_coords = np.vstack((np.cumsum(x1), np.cumsum(y1))).T

        angles[-2:] *= -1
        angle_sum = np.cumsum(angles)[1:]
        x2, y2 = np.zeros(len(angles)), np.zeros(len(angles))
        x2[1:], y2[1:] = lengths * np.cos(angle_sum), lengths * np.sin(angle_sum)
        x2y2 = np.vstack((np.cumsum(x2), np.cumsum(y2))).T[-3:]

        segments = np.vstack([arm_coords[-2:][::-1], x2y2])
        segments += [-2, 0]
        segments *= 0.7

        return segments

    def trajectories_map(self, wfile=None, ax=None, palette=None):
        if wfile is None:
            wfile = self.sm.site_dir / "trajectories.npy"
        data = np.load(wfile, allow_pickle=True)
        cells, stime, _ = data.shape
        side = int(np.sqrt(cells))
        fig = None if ax else plt.figure(figsize=(8, 8))
        colors = palette(np.linspace(0, 1, stime))
        for cell in range(cells):
            if fig:
                ax = fig.add_subplot(side, side, cell + 1, aspect="equal")
            ax.add_collection(
                LineCollection(
                    segments=np.hstack(
                        [
                            data[cell].reshape(-1, 1, 2)[:-1],
                            data[cell].reshape(-1, 1, 2)[1:],
                        ]
                    ),
                    colors=colors,
                )
            )
            ax.scatter(*data[cell].T, c=palette(np.linspace(0, 1, stime)), alpha=0.1)
            ax.set_xlim([-0.1, np.pi / 2 + 0.1])
            ax.set_ylim([-0.1, np.pi / 2 + 0.1])
            ax.set_xticks([])
            ax.set_yticks([])
        if fig:
            fig.tight_layout(pad=0.0)
            fig.savefig(self.sm.site_dir / "trajectories.png")

    def policy_map(self, wfile=None, ax=None):
        side = int(np.sqrt(self.params.internal_size))
        if wfile is None:
            wfile = self.sm.site_dir / "weights.npy"
        data = np.load(wfile, allow_pickle=True)[0]["policy"].T
        pca = PCA(n_components=3)
        data_pca = pca.fit_transform(data)
        data_pca = data_pca.reshape(internal_side, side, 3)
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(data_pca)
        ax.set_axis_off()
        if fig:
            fig.tight_layout(pad=0.0)
            dir_ = self.sm.site_dir if self.sm.site_dir.exists() else Path(".")
            fig.savefig(dir_ / "policy_map.png")

    def visual_map(self, wfile=None, ax=None):
        internal_side = int(np.sqrt(self.params.internal_size))
        visual_side = int(np.sqrt(self.params.visual_size // 3))

        if wfile is None:
            wfile = self.sm.site_dir / "weights.npy"
        data_v = np.load(wfile, allow_pickle=True)[0]["visual"]
        data_v = data_v.reshape(visual_side, visual_side, 3, internal_side, internal_side)
        data_v = data_v.transpose(3, 0, 4, 1, 2)
        data_v = data_v[::-1, :, :, :, :]
        data_v = data_v.reshape(
            visual_side * internal_side, visual_side * internal_side, 3
        )
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow((data_v - data_v.min()) / (data_v.max() - data_v.min()))
        ax.set_ylim(-0.5 * visual_side, (internal_side + 0.5) * visual_side)
        ax.set_ylim((internal_side + 0.5) * visual_side, -0.5 * visual_side)
        ax.set_axis_off()
        if fig:
            fig.tight_layout(pad=0.0)
            fig.savefig(self.sm.site_dir / "visual_map.png")
            plt.close("all")

    def proprio_map(self, wfile=None, ax=None):
        internal_side = int(np.sqrt(self.params.internal_size))
        if wfile is None:
            wfile = self.sm.site_dir / "weights.npy"
        data = np.load(wfile, allow_pickle=True)[0]["proprio"]
        ss_dim, _ = data.shape
        data = data.reshape(ss_dim, internal_side, internal_side)
        data = data.transpose(1, 2, 0)
        grips = []
        for j in range(internal_side):
            for i in range(internal_side):
                grip = self.generate_gripper(data[j, i][-2:])
                grip = (grip - [0, -1]) / [1, 2]
                grips.append(grip + [[i, j]])
        grips = np.stack(grips)
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        for grip in grips:
            ax.plot(*(grip).T, c="black", marker="o", markersize=1.5)
        ax.set_xlim(-0.5, internal_side + 0.5)
        ax.set_ylim(-0.5, internal_side + 0.5)
        ax.set_axis_off()
        if fig:
            fig.tight_layout(pad=0.0)
            fig.savefig(self.sm.site_dir / "proprio_map.png")
            plt.close("all")

    def somatosensory_map(self, wfile=None, ax=None):

        internal_side = int(np.sqrt(self.params.internal_size))
        if wfile is None:
            wfile = self.sm.site_dir / "weights.npy"
        data = np.load(wfile, allow_pickle=True)[0]["ssensory"]
        ss_dim, _ = data.shape
        data = data.reshape(ss_dim, internal_side, internal_side)
        data = data.transpose(1, 2, 0)
        data = data.reshape(internal_side * internal_side, ss_dim)
        data = np.clip(data, 0, None)
        grid = self.generate_sensor_grid()
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        for i in range(internal_side * internal_side):
            ax.scatter(*np.array(grid[i]).T, c="grey", s=0.05)
            ax.scatter(*np.array(grid[i]).T, c="black", s=2 * data[i])
        ax.set_xlim(-0.5, internal_side + 0.5)
        ax.set_ylim(-0.5, internal_side + 0.5)
        ax.set_axis_off()
        if fig:
            fig.tight_layout(pad=0.0)
            fig.savefig(self.sm.site_dir / "ssensory_map.png")
            plt.close("all")

    def comp_map(self, wfile=None, ax=None):
        internal_side = int(np.sqrt(self.params.internal_size))
        if wfile is None:
            wfile = self.sm.site_dir / "comp_grid.npy"
        data_c = np.load(wfile, allow_pickle=True)
        data_c = data_c.reshape(internal_side, internal_side)
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.imshow(data_c, vmin=0, vmax=1)
            ax.set_axis_off()
            fig.tight_layout(pad=0.0)
            fig.savefig(self.sm.site_dir / "comp_map.png")
            plt.close("all")
        else:
            ax.imshow(data_c, vmin=0, vmax=1)
            ax.set_axis_off()

    def goal_frequency_map(self, v_p_set):
        internal_side = int(np.sqrt(self.params.internal_size))
        data = np.zeros((internal_side, internal_side))
        for k in v_p_set:
            data[int(k[0]), int(k[1])] = v_p_set[k]
        data /= data.sum()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, aspect="equal")
        ax.imshow(data)
        ax.set_axis_off()
        fig.tight_layout(pad=0.0)
        fig.savefig(self.sm.site_dir / "goal_frequency_map.png")
        plt.close("all")

    def representations_movements(
        self,
        v_r,
        ss_r,
        p_r,
        a_r,
        name,
        vidManager=None,
    ):
        internal_size = self.params.internal_size
        internal_side = int(np.sqrt(self.params.internal_size))
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, aspect="equal")
        vm = vidManager(fig, "irep", "irep", duration=50)
        x = np.arange(internal_side)
        grid = np.stack(np.meshgrid(x, x)).reshape(2, -1)
        pv = ax.scatter(*grid, c="green", s=np.ones(internal_size))
        pss = ax.scatter(*grid, c="red", s=100 * np.ones(internal_size))
        pp = ax.scatter(*grid, c="blue", s=100 * np.ones(internal_size))
        pa = ax.scatter(*grid, c="black", s=100 * np.ones(internal_size))
        for i, (v, ss, p, a) in enumerate(zip(v_r, ss_r, p_r, a_r)):
            pv.set_sizes(700 * v)
            pss.set_sizes(700 * ss)
            pp.set_sizes(700 * p)
            pa.set_sizes(700 * a)
            ax.set_title("%d" % i)
            vm.save_frame()
        vm.mk_video(name=name, dirname=".")
        plt.close("all")

    def log(self, wfile=None):
        if wfile is None:
            wfile = self.sm.site_dir / "log.npy"
        log_data = np.load(wfile, allow_pickle=True)
        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(111)
        stime = len(log_data)
        ax.fill_between(
            np.arange(stime), log_data[:, 0], log_data[:, 2], fc="red", alpha=0.3
        )
        ax.plot(np.arange(stime), log_data[:, 1], c=[0.5, 0, 0])
        ax.set_xlim([-stime * 0.1, stime * 1.1])
        m = log_data.max()
        if m > 0:
            ax.set_ylim([-m * 0.1, m * 1.1])
        fig.savefig(self.sm.site_dir / "log.png")
        plt.close("all")


if __name__ == "__main__":

    wfile = "weights.npy"
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    axes = axes.T.flatten()
    for ax in axes:
        ax.set_axis_off()

    fig.tight_layout(pad=0)

    sm = StorageManager()
    params = Parameters()

    gm = GraphManager(sm, params)

    gm.visual_map(wfile, ax=axes[0])
    gm.proprio_map(wfile, ax=axes[1])
    gm.somatosensory_map(wfile, ax=axes[2])
    gm.policy_map(wfile, ax=axes[3])
