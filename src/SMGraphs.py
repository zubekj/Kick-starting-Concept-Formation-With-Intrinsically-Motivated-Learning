import glob
import os
import pathlib
from shutil import copyfile, rmtree

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from box2dsim.envs.mkvideo import vidManager
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from shapely import LineString, MultiLineString
from sklearn.decomposition import PCA

from params import Parameters


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

simulations_dir = "simulations"
storage_dir = "storage"
site_dir = "www"


def remove_figs(epoch=0):
    if epoch > 0:
        epoch_dir = f"{storage_dir}/{epoch:06d}"
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(site_dir, exist_ok=True)
        os.makedirs(epoch_dir, exist_ok=True)

        try:
            copyfile(f"{site_dir}/visual_map.png", f"{epoch_dir}/visual_map.png")
            copyfile(f"{site_dir}/comp_map.png", f"{epoch_dir}/comp_map.png")
            copyfile(f"{site_dir}/ssensory_map.png", f"{epoch_dir}/ssensory_map.png")
            copyfile(f"{site_dir}/proprio_map.png", f"{epoch_dir}/proprio_map.png")
            copyfile(f"{site_dir}/log.png", f"{epoch_dir}/log.png")
            copyfile(
                f"{site_dir}/goal_frequency_map.png",
                f"{epoch_dir}/goal_frequency_map.png",
            )
        except OSError:
            pass
    else:
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(site_dir, exist_ok=True)
        print("Starting simulation ...")
        if not os.path.exists(f"{site_dir}/blank.gif"):
            blank_video()
        os.makedirs("storage", exist_ok=True)
        copyfile(f"{site_dir}/blank.gif", f"{site_dir}/tv.gif")
        copyfile(
            f"{pathlib.Path(__file__).parent.resolve()}/arms.html",
            f"{site_dir}/arms.html",
        )

        figs = glob.glob(f"{site_dir}/episode*.gif") + glob.glob(f"{site_dir}/*.png")
        for f in figs:
            if os.path.isdir(f):
                rmtree(f)
            else:
                os.remove(f)
        for k in range(params.tests):
            copyfile(f"{site_dir}/blank.gif", f"{site_dir}/episode_{k}_demo.gif")

        copyfile(f"{site_dir}/blank.gif", f"{site_dir}/visual_map.png")
        copyfile(f"{site_dir}/blank.gif", f"{site_dir}/comp_map.png")
        copyfile(f"{site_dir}/blank.gif", f"{site_dir}/log.png")


def update_weight_data(weights=None, tag=None, epoch=None):

    if weights is None:
        storage_dir = (
            f"storage{'-' if tag is not None else '' }{tag if tag is not None else ''}"
        )
        if os.path.isdir(storage_dir):
            if epoch is None:
                epochs = sorted(glob.glob(f"{storage_dir}/*"))
                epoch_dir = f"{epochs[-1]}"
            else:
                epoch_dir = f"{storage_dir}/{epoch:06d}"

            weights = np.load(
                f"{epoch_dir}/weights.npy",
                allow_pickle=True,
            )[0]
        else:
            raise Exception(f"{storage_dir} does not exist!")

    np.save(f"{site_dir}/weights", weights)


def generate_sensor_points(n_sensors):
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

    line = MultiLineString(
        [LineString(opoints[i]) for i in range(8)],
    )
    points = [
        [i.x, i.y] for i in line.interpolate(np.linspace(0, line.length, n_sensors))
    ]
    points = np.array(points)

    def normalize_array(arr):
        norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return norm_arr

    points[:, 0] = normalize_array(points[:, 0])
    points[:, 1] = normalize_array(points[:, 1])

    points *= 0.8
    points += 0.1
    return points


def generate_sensor_grid(side, n_sensors):
    grid = []
    for i in range(side):
        for j in range(side):
            points = generate_sensor_points(n_sensors)
            grid.append(points + [j, i])
    grid = np.array(grid)
    grid.reshape(side * side, n_sensors, 2)

    return grid


def trajectories_map(wfile=None, ax=None):
    if wfile is None:
        wfile = f"{site_dir}/trajectories.npy"
    data = np.load(wfile, allow_pickle=True)
    cells, stime, _ = data.shape
    side = int(np.sqrt(cells))
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
    colors = palette(np.linspace(0, 1, stime))
    for cell in range(cells):
        if ax is None:
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

    if ax is None:
        fig.tight_layout(pad=0.0)
        fig.savefig(f"{site_dir}/trajectories.png")


def policy_map(wfile=None, ax=None):
    if wfile is None:
        wfile = f"{site_dir}/weights.npy"
    data = np.load(wfile, allow_pickle=True)[0]["policy"].T
    pca = PCA(n_components=3)
    data_pca = pca.fit_transform(data)

    data_pca = data_pca.reshape(internal_side, internal_side, 3)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.imshow(data_pca)
    ax.set_axis_off()
    if ax is None:
        fig.tight_layout(pad=0.0)
    if not os.path.exists(site_dir):
        dir_ = "."
    else:
        dir_ = site_dir
    if ax is None:
        fig.savefig(f"{dir_}/policy_map.png")


def visual_map(wfile=None, ax=None):
    if wfile is None:
        wfile = f"{site_dir}/weights.npy"
    data_v = np.load(wfile, allow_pickle=True)[0]["visual"]
    data_v = data_v.reshape(visual_side, visual_side, 3, internal_side, internal_side)
    data_v = data_v.transpose(3, 0, 4, 1, 2)
    data_v = data_v[::-1, :, :, :, :]
    data_v = data_v.reshape(visual_side * internal_side, visual_side * internal_side, 3)

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
    ax.imshow((data_v - data_v.min()) / (data_v.max() - data_v.min()))
    ax.set_axis_off()
    if ax is None:
        fig.tight_layout(pad=0.0)
        fig.savefig(f"{site_dir}/visual_map.png")
        plt.close("all")


def proprio_map(wfile=None, ax=None):
    if wfile is None:
        wfile = f"{site_dir}/weights.npy"

    data = np.load(wfile, allow_pickle=True)[0]["proprio"]
    ss_dim, _ = data.shape
    data = data.reshape(ss_dim, internal_side, internal_side)
    data = data.transpose(1, 2, 0)

    grips = []
    for j in range(internal_side):
        for i in range(internal_side):
            grip = generate_gripper(data[j, i][-2:])
            grips.append(grip + [[i, j]])
    grips = np.stack(grips)

    if ax is None:
        fig, ax = plt.subplots()

    for grip in grips:
        ax.plot(*grip.T, c="black")
    ax.set_axis_off()

    if ax is None:
        fig.tight_layout(pad=0.0)
        fig.savefig(f"{site_dir}/proprio_map.png")
        plt.close("all")


#####
def somatosensory_map(wfile=None, ax=None):
    if wfile is None:
        wfile = f"{site_dir}/weights.npy"

    data = np.load(wfile, allow_pickle=True)[0]["ssensory"]
    ss_dim, _ = data.shape
    data = data.reshape(ss_dim, internal_side, internal_side)
    data = data.transpose(1, 2, 0)
    data = data.reshape(internal_side * internal_side, ss_dim)

    grid = generate_sensor_grid(internal_side, ss_dim)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    for i in range(internal_side * internal_side):
        ax.scatter(
            *np.array(grid[i]).T,
            c="grey",
            s=0.05,
        )
        ax.scatter(
            *np.array(grid[i]).T,
            c="black",
            s=2 * data[i],
        )
    ax.set_axis_off()
    if ax is None:
        fig.tight_layout(pad=0.0)
        fig.savefig(f"{site_dir}/ssensory_map.png")
        plt.close("all")


#####


def comp_map(wfile=None, ax=None):
    if wfile is None:
        wfile = f"{site_dir}/comp_grid.npy"
    data_c = np.load(wfile, allow_pickle=True)
    data_c = data_c.reshape(internal_side, internal_side)

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.imshow(data_c, vmin=0, vmax=1)
        ax.set_axis_off()
        fig.tight_layout(pad=0.0)
        fig.savefig(f"{site_dir}/comp_map.png")
        plt.close("all")
    else:
        ax.imshow(data_c, vmin=0, vmax=1)
        ax.set_axis_off()


def goal_frequency_map(v_p_set):

    data = np.zeros((internal_side, internal_side))

    for k in v_p_set:
        data[int(k[0]), int(k[1])] = v_p_set[k]

    data /= data.sum()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    ax.imshow(data)
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(f"{site_dir}/goal_frequency_map.png")
    plt.close("all")


def generate_gripper(angles):

    gsegment = 0.25
    segments = np.ones([4, 2, 2]) * [[[0, 0.25]]]

    for i, angle in enumerate(np.cumsum(angles)):
        segments[i][1] = segments[i][0] + gsegment * np.array(
            [np.cos(angle), np.sin(angle)]
        )
        if i == 0:
            segments[i + 1][0] = np.copy(segments[i][1])

    segments = np.vstack([segments * [[[-1, 1]]], segments]) + [[[0.5, 0]]]

    segments = [[[0, 1]]] - segments

    return segments


def representations_movements(v_r, ss_r, p_r, a_r, name):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    vm = vidManager(fig, "irep", "irep", duration=50)

    x = np.arange(internal_side)
    grid = np.stack(np.meshgrid(x, x)).reshape(2, -1)

    pv = ax.scatter(*grid, c="green", s=np.ones(params.internal_size))
    pss = ax.scatter(*grid, c="red", s=100 * np.ones(params.internal_size))
    pp = ax.scatter(*grid, c="blue", s=100 * np.ones(params.internal_size))
    pa = ax.scatter(*grid, c="black", s=100 * np.ones(params.internal_size))

    for i, (v, ss, p, a) in enumerate(zip(v_r, ss_r, p_r, a_r)):
        pv.set_sizes(700 * v)
        pss.set_sizes(700 * ss)
        pp.set_sizes(700 * p)
        pa.set_sizes(700 * a)
        ax.set_title("%d" % i)
        vm.save_frame()

    vm.mk_video(name=name, dirname=".")
    plt.close("all")


def blank_video():
    name = "blank"
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect="equal")
    vm = vidManager(fig, "blank", f"{site_dir}/blank", duration=50)

    ax.set_visible(False)

    for t in range(5):
        vm.save_frame()

    vm.mk_video(name=name, dirname=f"{site_dir}")
    plt.close("all")


def log(wfile=None):
    if wfile is None:
        wfile = f"{site_dir}/log.npy"
    log = np.load(wfile, allow_pickle=True)
    fig = plt.figure(figsize=(4, 2))
    ax = fig.add_subplot(111)
    stime = len(log)
    ax.fill_between(np.arange(stime), log[:, 0], log[:, 2], fc="red", alpha=0.3)
    ax.plot(np.arange(stime), log[:, 1], c=[0.5, 0, 0])
    ax.set_xlim([-stime * 0.1, stime * 1.1])
    m = log.max()
    if m > 0:
        ax.set_ylim([-m * 0.1, m * 1.1])
    fig.savefig(f"{site_dir}/log.png")
    plt.close("all")


if __name__ == "__main__":

    wfile = "weights.npy"
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    axes = axes.T.flatten()
    for ax in axes:
        ax.set_axis_off()

    fig.tight_layout(pad=0)

    visual_map(wfile, ax=axes[0])
    proprio_map(wfile, ax=axes[1])
    somatosensory_map(wfile, ax=axes[2])
    policy_map(wfile, ax=axes[3])
