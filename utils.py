"""Utility functions."""

import random

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from pylab import rcParams

plt.style.use('classic')
rcParams['figure.figsize'] = 3, 3


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def plot_mdn_density(model, hx, D, device, output_file):
    model.eval()
    plt.clf()
    fig = plt.figure(figsize=(15, 4))

    grid_size = 200
    x, y = torch.meshgrid(torch.linspace(-3, 3, grid_size),
                          torch.linspace(-3, 3, grid_size))
    data = torch.stack((x, y), dim=-1).view(-1, 2)

    hx_ = hx.repeat(data.shape[0], 1)
    yy = model(hx_.to(device))
    llh = model.log_likelihood(yy, data.to(device))
    llh = llh.view(grid_size, grid_size).exp().data.cpu().numpy()

    plt.subplot(1, 2, 1)
    plt.pcolormesh(x, y, llh, cmap='jet', vmin=0)
    plt.title('estimated mixture density')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.hist2d(D[:, 0], D[:, 1], bins=200, range=((-3, 3), (-3, 3)))
    plt.title('data density')
    plt.axis('equal')

    plt.savefig(output_file)
    plt.close()
