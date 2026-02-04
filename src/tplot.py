# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.objects as so
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class TPlotManager:
    """
    A class for managing and plotting the trajectories of postures during demo
    trials of multiple prototypes on a grid of subplots.

    Attributes:
        n_prototypes (int): Number of prototype plots to manage and display,
            determining the grid's total number of plots.
        limits (tuple): Axis limits for each subplot, defining the range of
            data visualized on the x and y axes.
        figsize (tuple): Size of the entire figure layout, affecting the
            overall visualization's dimensions.
        max_ts (float): Maximum value for the time series data, used for
            plotting purposes.

    Methods:
        plot_prototype(axis, data):
            Plots individual prototype data on the specified axes within the
            grid. Supports rendering of trajectory data, where each data point
            can be color-coded according to a time series scale.
    """

    def __init__(
        self,
        n_prototypes=100,
        figsize=(5, 4),
        max_ts=100,
        plot_path="trajectory_plots.png",
    ):
        """Initializes the TPlotManager with given parameters.

        Args:
            n_prototypes (int): Number of prototypes to plot.
            figsize (tuple): Size of the figure.
            max_ts (int): Maximum time series value.
        """
        self._n_prototypes = n_prototypes
        self._side = int(np.sqrt((self._n_prototypes)))
        self._fig, self._axes = plt.subplots(self._side, self._side, figsize=figsize)
        self._axes = self._axes.flatten()
        self._max_ts = max_ts
        self.plot_path = plot_path

        # Configure each axis: turn off axis, set x and y limits
        for ax in self._axes:

            ax.set_xticks([])
            ax.set_yticks([])

        self._fig.tight_layout(pad=0)
        self.plots = {}

    def plot_prototype(self, data, i):
        """Plots the posture trajectory for a prototype on the specified axis.

        Args:
            i (int): Index of the axis to plot on.
            data (DataFrame): Data to plot, containing fields:
                - x: X-axis values.
                - y: Y-axis values.
                - ts: Color scale values for dots.
        """
        self.plots[i] = (
            so.Plot(data, x="x", y="y")
            .add(so.Dot(pointsize=3), color="ts", legend=False)
            .add(so.Path(color="black", linewidth=0.6, alpha=0.4))
            .scale(color="YlOrBr")
            .on(self._axes[i])
            .plot()
        )
        self._axes[i].set_xticks([])
        self._axes[i].set_yticks([])
        self._axes[i].set_xlabel("")
        self._axes[i].set_ylabel("")

    def plot_prototypes(self, data):
        """Plots the posture trajectory for a set of prototypes.

        Args:
            data (DataFrame): Data to plot, containing fields:
                tr_id (int): the current trajectory
                x (float): X-axis values.
                y (float): Y-axis values.
                prototype (int): Index of the prototype for this trajectory
                    measure.
                ts (float): Color scale values for dots.
        """
        data = self.reduce_dimensions_with_pca(data)

        # data[["x", "y"]] = data[["d1", "d2"]]

        data.loc[:, "x"] = (data.x - data.x.min()) / np.ptp(data.x)
        data.loc[:, "y"] = (data.y - data.y.min()) / np.ptp(data.y)

        for ax in self._axes:
            ax.set_xlim(0, np.ptp(data.x))
            ax.set_ylim(0, np.ptp(data.y))

        prototype_set = set()
        for i, prototype in data.groupby("prototype"):
            if i not in prototype_set:
                self.plot_prototype(prototype, int(i))
                prototype_set.add(i)
        plt.savefig(self.plot_path)

    def reduce_dimensions_with_pca(self, data):
        """Reduces the dimensionality of the postures in trajectories to 2D.

        This function takes a DataFrame containing posture data and reduces
        its dimensionality to two dimensions for visualization purposes.

        Args:
            data (DataFrame): The data to plot, which includes the following
                fields:
                - d1 (float): Values in the first dimension (1st joint angle).
                - d2 (float): Values in the second dimension (1st joint angle).
                - prototype (int): Index of the prototype for this trajectory
                  measure.
                - ts (float): Color scale values for dots.
        """
        pca = PCA(n_components=2)
        scaler = StandardScaler()
        dim_columns = [d for d in data.columns if "d" in d]
        data.loc[:, ["x", "y"]] = pca.fit_transform(
            scaler.fit_transform(data[dim_columns])
        )
        return data


def generate_demo_prototype_data(n_ts, n_prototypes):
    df = []
    for i in range(n_prototypes):
        if np.random.rand() > 0.0:
            direction = (0.4 + 0.2 * np.random.randn()) * 0.5 * np.pi
            curvature = 0.1 * np.random.randn()
            x = np.linspace(0, 30 * np.cos(direction), n_ts)
            curvature = 0.01 * np.random.randn()
            y = np.linspace(0, 30 * np.sin(direction), n_ts) + curvature * x**2
            curvature = 0.01 * np.random.randn()
            z = np.linspace(0, 30 * np.sin(direction), n_ts) * 0.5 + curvature * x**2
            ts = np.arange(n_ts)
            data = pd.DataFrame({"d1": x, "d2": y, "d3": z, "ts": ts, "prototype": i})
            df.append(data)
    return pd.concat(df)


# %%

if __name__ == "__main__":
    # %%
    n_ts = 148
    n_prototypes = 100

    tp = TPlotManager(n_prototypes=n_prototypes)
    # %%

    print("Generate demo data ...")
    df = pd.read_csv("trajectories.csv")
    df.loc[:, "prototype"] = df.prototype_x * int(np.sqrt(n_prototypes)) + df.prototype_y
    df = df.groupby(["prototype", "ts"]).mean().reset_index()

    # %%
    print("Plot postures ...")
    tp.plot_prototypes(df)
