"""Training script."""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from dataset import MixtureDataset
from logger import Logger
from model import MDN
from train import train_one_epoch
from utils import plot_mdn_density, set_seed


def main():
    parser = ArgumentParser()
    parser.add_argument('--num-mixtures', default=10, type=int)
    parser.add_argument('--num-data-mixtures', default=5, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--num-epochs', default=10, type=int)
    parser.add_argument('--hidden-dim', default=128, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    print('Config', args._get_kwargs())

    dataset = MixtureDataset(num_mixtures=args.num_data_mixtures)
    set_seed(args.seed)
    data_loader = DataLoader(dataset.data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MDN(hidden_dim=args.hidden_dim, num_mixtures=args.num_mixtures)
    model = model.to(device)
    hx = torch.randn(1, args.hidden_dim)
    hx_ = hx.repeat(args.batch_size, 1).to(device).data

    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    lr_scheduler = OneCycleLR(optimizer=optimizer,
                              max_lr=args.learning_rate,
                              epochs=args.num_epochs,
                              steps_per_epoch=len(data_loader))

    logger = Logger('./log')

    for epoch in range(args.num_epochs):
        train_one_epoch(epoch, model, data_loader, optimizer, lr_scheduler,
                        device, hx_, logger)
        logger.plot(f'log_{epoch}.png')
        plot_mdn_density(model, hx, dataset.data, device,
                         f'log/density_{epoch}.png')

    pred = torch.chunk(model(hx), chunks=6, dim=-1)
    weight = pred[-1]
    plt.clf()
    plt.matshow(weight.softmax(dim=-1).data.cpu().numpy(), cmap='hot')
    plt.colorbar()
    plt.savefig('./log/weight.png')


if __name__ == '__main__':
    main()
