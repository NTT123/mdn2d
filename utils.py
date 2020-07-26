"""Utility functions."""


import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pylab import rcParams
import random

plt.style.use('classic')
rcParams['figure.figsize'] = 3, 3


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def plot_mdn_density(model, hx, D, device, output_file):
    plt.clf()
    plt.close()
    y, x = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    tx = torch.tensor(x)
    ty = torch.tensor(y)
    ii = torch.stack((tx, ty), dim=-1)
    iii = ii.float().view(-1, 2)
    xx = hx[0:1].repeat(iii.size(0), 1)
    model.eval()
    yy = model(xx.to(device))
    p = model.log_likelihood(yy, iii.to(device))
    z = p.view(200, 200).exp().double().data.cpu().numpy()
    z = z[:-1, :-1]
    z_min, z_max = 0., np.abs(z).max()
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    c = ax.pcolormesh(x, y, z, cmap='jet', vmin=z_min, vmax=z_max)
    ax.set_title('estimated mixture density')
    plt.axis('equal')
    ax.axis([x.min(), x.max(), y.min(), y.max()])

    plt.subplot(1, 2, 2)
    _ = plt.hist2d(D[:, 0], D[:, 1], bins=200, range=((-3, 3), (-3, 3)))
    ax = plt.gca()
    ax.set_title('data density')
    plt.axis('equal')
    ax.axis([x.min(), x.max(), y.min(), y.max()])

    plt.savefig(output_file)
