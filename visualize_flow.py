import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import math
from matplotlib import cm



LOW = -4
HIGH = 4


def plt_potential_func(potential, ax, npts=100, title="$p(x)$"):
    """
    Args:
        potential: computes U(z_k) given z_k
    """
    xside = np.linspace(LOW, HIGH, npts)
    yside = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(xside, yside)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.Tensor(z)
    u = potential(z).cpu().numpy()
    p = np.exp(-u).reshape(npts, npts)

    plt.pcolormesh(xx, yy, p)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow(prior_logdensity, transform, ax, npts=100, title="$q(x)$", device="cpu"):
    """
    Args:
        transform: computes z_k and log(q_k) given z_0
    """
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.tensor(z, requires_grad=True).type(torch.float32).to(device)
    logqz = prior_logdensity(z)
    logqz = torch.sum(logqz, dim=1)[:, None]
    z, logqz = transform(z, logqz)
    logqz = torch.sum(logqz, dim=1)[:, None]

    xx = z[:, 0].cpu().numpy().reshape(npts, npts)
    yy = z[:, 1].cpu().numpy().reshape(npts, npts)
    qz = np.exp(logqz.cpu().numpy()).reshape(npts, npts)

    plt.pcolormesh(xx, yy, qz)
    ax.set_xlim(LOW, HIGH)
    ax.set_ylim(LOW, HIGH)
    cmap = matplotlib.cm.get_cmap(None)
    ax.set_facecolor(cmap(0.))
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_density(logdensity, ax, npts=100, memory=100, title="$q(x)$", device="cpu", low=LOW, high=HIGH):
    side = np.linspace(low, high, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)

    logpx = logdensity(x)
    logpx = logpx - logpx.max()

    px = np.exp(logpx.cpu().detach().numpy()).reshape(npts, npts)
    px = px / px.sum()

    ax.imshow(px, cmap=cm.viridis)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    #ax.set_title(title)


def plt_flow_density3d(logdensity, ax, npts=100, memory=100, title="$q(x)$", device="cpu", low=LOW, high=HIGH):
    side = np.linspace(low, high, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)

    logpx = logdensity(x)
    logpx = logpx - logpx.max()

    px = np.exp(logpx.cpu().detach().numpy()).reshape(npts, npts)
    px = px / px.sum()

    ax = plt.axes(projection='3d')
    #ax.contour3D(xx, yy, px, 50, cmap='viridis')
    ax.plot_surface(xx, yy, px, rstride=1, cstride=1, cmap='viridis')#, edgecolor='none')



def plt_flow_samples(prior_sample, transform, ax, npts=100, memory=100, title="$x ~ q(x)$", device="cpu"):
    z = prior_sample(npts * npts, 2).type(torch.float32).to(device)
    zk = []
    inds = torch.arange(0, z.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):
        zk.append(transform(z[ii]))
    zk = torch.cat(zk, 0).cpu().numpy()
    ax.hist2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)
    ax.set_aspect('equal')


def plt_samples(samples, ax, npts=100, title="$x ~ p(x)$", low=LOW, high=HIGH):
    ax.hist2d(samples[:, 0], samples[:, 1], range=[[low, high], [low, high]], bins=npts)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def visualize_transform(samples, sample_names, logdensities, logdensity_names,
    npts=100, memory=100, device="cpu", low=LOW, high=HIGH):
    """Produces visualization for the model density and samples from the model."""
    n_samples = len(samples)
    n_logdensities = len(logdensities)
    assert n_samples == len(sample_names)
    assert n_logdensities == len(logdensity_names)
    n_col = max(n_samples, n_logdensities)
    if n_samples > 0:
        for i, name in zip(range(n_samples), sample_names):
            if i == 0:
                plt.clf()
            ax = plt.subplot(2 if len(logdensities) > 0 else 1, n_col, i + 1, aspect="equal")
            ax.set_xlim(LOW, HIGH)
            ax.set_ylim(LOW, HIGH)
            plt_samples(samples[i], ax, npts=npts, title=name, low=low, high=high)

    if n_logdensities > 0:
        for i, name in zip(range(n_logdensities), logdensity_names):
            ax = plt.subplot(2 if len(samples) > 0 else 1, n_col,
                             n_col + i + 1 if len(samples) > 0 else i + 1, aspect="equal")
            plt_flow_density(logdensities[i], ax, npts=npts, memory=memory, device=device, title=name,
                             low=low, high=high)


def visualize_samples(samples, sample_names, n_col,
    npts=100, memory=100, device="cpu", low=LOW, high=HIGH):
    """Produces visualization for the model density and samples from the model."""
    n_samples = len(samples)
    n_row = n_samples // n_col
    for i, name in zip(range(n_samples), sample_names):
        if i == 0:
            plt.clf()
        ax = plt.subplot(n_row, n_col, i + 1, aspect="equal")
        plt_samples(samples[i], ax, npts=npts, title=name)


def visualize_slices(samples, sample_names,
    npts=100, memory=100, device="cpu", low=LOW, high=HIGH):
    """
    Produces visualization for the model density and samples from the model
    split by 2D slices each in a separate row.
    """

    assert samples.shape[1] % 2 == 0
    n_cols = len(samples)  # number of columns
    n_rows = int(samples.shape[1] / 2)  # number of rows
    for i, name in zip(range(n_rows), sample_names):
        if i == 0:
            plt.clf()
        for j in range(n_cols):
            ax = plt.subplot(n_rows, n_cols, i * j + j + 1, aspect="equal")
            plt_samples(samples[:, 2*j:2*j+2], ax, npts=npts, title=name)
