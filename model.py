import math

import torch
from torch import nn


class MDN(torch.nn.Module):
    """2d mixture density network."""

    def __init__(self, hidden_dim=128, num_mixtures=5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures

        ## dx, dy, logstdx, logstdy, angle, weight
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = torch.nn.Linear(128, num_mixtures * 6, bias=False)

    def log_likelihood(self, x, y):
        dx, dy, logstdx, logstdy, angle, weight = torch.chunk(x, 6, dim=-1)
        dx_data, dy_data = torch.chunk(y, 2, dim=-1)
        u = dx_data - dx
        v = dy_data - dy
        rotated_u = torch.cos(angle) * u - torch.sin(angle) * v
        rotated_v = torch.sin(angle) * u + torch.cos(angle) * v
        scaled_u = rotated_u / logstdx.exp()
        scaled_v = rotated_v / logstdy.exp()

        llh = -(0.5 * torch.pow(scaled_u, 2) +
                0.5 * torch.pow(scaled_v, 2) +
                logstdx + logstdy + math.log(math.pi * 2))

        weight = torch.nn.functional.log_softmax(weight, dim=-1)
        llh = llh + weight
        llh = torch.logsumexp(llh, dim=-1)

        return llh

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
