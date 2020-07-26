"""Generate a synthesized dataset of mixture gaussian distributions."""

import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.distributions import MultivariateNormal

DATA_DIRNAME = Path('./data')


class MixtureDataset:
    """Mixture dataset class."""

    def __init__(self, num_mixtures):
        self.num_mixtures = num_mixtures
        DATA_FILENAME = DATA_DIRNAME / f'mix_{num_mixtures}.hdf5'

        if not DATA_FILENAME.exists():
            print("Generating synthesized dataset...")
            DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
            data = _generate_data(100_000, num_mixtures)
            with h5py.File(DATA_FILENAME, "w") as f:
                f.create_dataset('mixture', data=data)
        with h5py.File(DATA_FILENAME, "r") as f:
            self.data = f['mixture'][:]


def _generate_data(N, num_mixtures):
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    weights = (0.2 * torch.rand(num_mixtures)).softmax(dim=0)
    means = 20 * (torch.rand(num_mixtures, 2) - 0.5)
    covs = [torch.randn(2, 2) for _ in range(num_mixtures)]
    covs = [torch.matmul(A, A.t()) + 0.1 * torch.eye(2) for A in covs]
    Zs = [MultivariateNormal(loc=means[c], covariance_matrix=covs[c])
          for c in range(num_mixtures)]

    # Generate data
    choices = np.random.choice(range(num_mixtures), size=N, p=weights.numpy())
    D = torch.empty((N, 2))

    for id, c in enumerate(choices):
        D[id, :] = Zs[c].sample()

    # zero mean, 1 std
    D = (D - D.mean(dim=0)) / D.std(dim=0)
    return D
