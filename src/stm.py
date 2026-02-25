# -*- coding: utf-8 -*-

"""
Guided Topological Map

An organizing map gtmp whose topology can be guided by a teaching signal.
Generalizes SOMs.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def is_2d(func):
    func.type = "2d"
    return func


def is_1d(func):
    func.type = "1d"
    return func


@is_1d
def radial(mean, sigma, size):
    """Gives radial bases on a 2D space, given a list of means and a std_dev.

    Args:
        mean (list): vector of means of radians
        sigma (float): standard deviation of radiants

    Returns:
        torch.Tensor: a radial basis of the input
    """

    x = torch.arange(size, dtype=torch.float)
    diff = x.view(1, -1) - mean.view(-1, 1)
    radial_basis = torch.exp(-0.5 * (sigma**-2) * (diff**2))
    return radial_basis


@is_2d
def radial2d(mean, sigma, size):
    """Gives radial bases on a 2D space, flattened into 1d vectors,
    given a list of means and a std_dev.

    Eg.

    size: 64              ┌────────┐         ┌─────────────────────────────────────────────────────────────────┐
    mean: (5, 4)  ----->  │........│  -----> │ ....................,+,....,oOo,...+O@O+...,oOo,....,+,.........│
    sigma: 1.5            │........│         └─────────────────────────────────────────────────────────────────┘
                          │....,+,.│
                          │...,oOo,│
                          │...+O@O+│
                          │...,oOo,│
                          │....,+,.│
                          │........│
                          └────────┘


    Args:
        mean (list): vector of means of radians
        sigma (float): standard deviation of radiants
        size (int): dimension of the flattened gaussian (side is sqrt(size))

    Returns:
        torch.Tensor: each row is a flattened radial basis in
                 the (sqrt(size), sqrt(size)) space
    """
    grid_points = make_grid(np.sqrt(size))
    diff = grid_points.unsqueeze(0) - mean.unsqueeze(1)
    radial_basis = torch.exp(-0.5 * (sigma**-2) * torch.norm(diff, dim=-1) ** 2)

    return radial_basis


def make_grid(side):
    """Creates a grid of points in 2D space."""
    ranges = torch.arange(0, side, dtype=torch.float)
    return torch.cartesian_prod(ranges, ranges)


class STM(torch.nn.Module):
    """A generic topological map"""

    def __init__(
        self,
        input_size,
        output_size,
        sigma,
        learn_intrinsic_distances=True,
        extrinsic_distances=None,
        radial_fun=radial2d,
        external_radial_prop=0.2,
        weights_init_sigma=1.0,
        **kwargs,
    ):
        """
        Args:
            output_size (int): number of elements in the output layer (shape will be
                                (sqrt(output_size), sqrt(output_size)))
            sigma (float): starting value of the extension of the learning neighbourhood
                           in the output layer.
            learn_intrinsic_distances (boolean): if learning depends on the distance of prototypes from inputs.
            exrinsic_distances (Tensor): if learning depends on the distance of prototypes from targets.

        """
        super(STM, self).__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.sigma = sigma
        self.learn_intrinsic_distances = learn_intrinsic_distances
        self.extrinsic_distances = extrinsic_distances
        self.radial_fun = radial_fun
        self.side = np.sqrt(output_size)
        self.grid = make_grid(self.side)
        self.grid = self.grid.unsqueeze(0)
        self.internal_radials = None
        self.external_radials = None
        self.external_radial_prop = external_radial_prop

        self.kernel = torch.nn.Parameter(
            torch.zeros(input_size, output_size), requires_grad=True
        )
        torch.nn.init.xavier_normal_(self.kernel, gain=weights_init_sigma)

    def forward(self, x):
        norms2, radials = self.get_norms_and_activation(x)
        return radials * norms2

    def spread(self, x):
        _, radials = self.get_norms_and_activation(x)
        radials = radials / (torch.sum(radials, dim=1).reshape(-1, 1) + 1e-5)
        return radials

    def get_norms_and_activation(self, x):
        # compute norms
        norms = torch.norm(x.unsqueeze(-1) - self.kernel.unsqueeze(0), dim=1)
        norms2 = norms.pow(2)

        # compute activation
        wta = norms2.argmin(dim=-1).detach().float()
        wta = torch.stack((wta // self.side, wta % self.side)).T
        radials = self.radial_fun(wta, self.sigma, self.output_size)
        radials = radials / (self.sigma * np.sqrt(np.pi * 2))

        self.internal_radials = radials

        if self.external_radials is not None:
            radials = (
                1 - self.external_radial_prop
            ) * radials + self.external_radial_prop * self.external_radials

        return norms2, radials

    def backward(self, radials):
        radials = radials / (radials.sum(dim=1).reshape(-1, 1) + 1e-5)

        # CHANGE: Use single prototype instead of weighted average.
        # x = torch.matmul(radials, self.kernel.T)
        x = self.kernel.T[radials.argmax(dim=-1)]

        return x

    def loss(self, radial_norms2, extrinsic=None):
        if extrinsic is None:
            extrinsic = torch.ones_like(radial_norms2)
        return torch.mean(radial_norms2 * extrinsic)

    def get_radials(self):
        return self.internal_radials

    def set_radials(self, radials):
        self.external_radials = radials


class SMSTM(STM):

    def __init__(
        self,
        learning_rate=2.0,
        **kwargs,
    ):
        super(SMSTM, self).__init__(**kwargs)
        self.lr = learning_rate
        self.modulate = 1.0
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def spread(self, x):
        return (
            super(SMSTM, self)
            .spread(torch.tensor(x, dtype=torch.float))
            .cpu()
            .detach()
            .numpy()
        )

    def get_activation(self, x):
        activation = super(SMSTM, self).forward(torch.tensor(x, dtype=torch.float))
        return activation.cpu().detach().numpy()

    def backward(self, x):
        return (
            super(SMSTM, self)
            .backward(torch.tensor(x, dtype=torch.float))
            .cpu()
            .detach()
            .numpy()
        )

    def get_point_and_representation(self, out, sigma=None):
        out = torch.tensor(out, dtype=torch.float)

        if sigma is None:
            sigma = self.sigma

        idx = torch.argmax(out, dim=1).detach().float()
        point = torch.stack((idx // self.side, idx % self.side)).T

        radials = self.radial_fun(point, sigma, self.output_size)
        radials = radials / (sigma * np.sqrt(np.pi * 2))
        radials = radials / (torch.sum(radials, dim=1).reshape(-1, 1) + 1e-5)

        return point.cpu().numpy(), radials.cpu().numpy()

    def getRepresentation(self, point, sigma=None):
        if sigma is None:
            sigma = self.sigma

        point = torch.tensor(point, dtype=torch.float)
        radials = self.radial_fun(point, sigma, self.output_size)
        radials = radials / (sigma * np.sqrt(np.pi * 2))
        radials = radials / (torch.sum(radials, dim=1).reshape(-1, 1) + 1e-5)

        return radials.cpu().numpy()

    def update_params(self, sigma=None, lr=None):
        if sigma is not None:
            self.sigma = torch.tensor(sigma)
        if lr is not None:
            self.modulate = lr

    def update(self, data, dists):
        assert len(data.shape) == 2
        assert data.shape[1] == self.input_size
        data = torch.tensor(data, dtype=torch.float)
        dists = torch.tensor(dists, dtype=torch.float)
        self.optimizer.zero_grad()
        out = self(data)
        loss = self.loss(out, dists) * self.modulate
        loss.backward()
        self.optimizer.step()
        return loss

    def set_weights(self, weights):
        with torch.no_grad():
            self.kernel.copy_(torch.tensor(weights, dtype=torch.float))

    def get_weights(self):
        return self.kernel.detach().cpu().numpy()


if __name__ == "__main__":

    inp_num = 2
    out_num = 100
    initial_sigma = out_num / 2
    min_sigma = 1
    initial_lr = 0.1
    stime = 10000
    decay_window = stime / 8

    loss = []

    som_layer = STM(inp_num, out_num, initial_sigma, radial_fun=radial2d)
    optimizer = torch.optim.Adam(som_layer.parameters(), lr=initial_lr)

    for t in range(stime):
        # learning rate and sigma annealing
        curr_sigma = min_sigma + initial_sigma * np.exp(-t / decay_window)
        curr_rl = initial_lr * np.exp(-t / decay_window)

        # update learning rate and sigma in the graph
        som_layer.sigma = curr_sigma
        optimizer.param_groups[0]["lr"] = curr_rl
        optimizer.zero_grad()

        data = torch.tensor(np.random.uniform(0, 1, [100, 2]))
        output = som_layer(data)
        loss_ = som_layer.loss(output)
        loss_.backward()

        optimizer.step()
        if t % (stime // 10) == 0:
            print(loss_.detach().numpy(), curr_sigma, curr_rl)
        loss.append(loss_)

    matplotlib.use("Agg")

    weights = som_layer.kernel.detach().numpy()
    plt.scatter(*weights)
    plt.savefig("weights.png")
