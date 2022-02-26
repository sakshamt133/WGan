import torch.nn as nn
import torch
from block import UpSample


class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            UpSample(noise_dim, 128, (3, 3)),
            UpSample(128, 100, (3, 3), (1, 1)),
            UpSample(100, 64, (3, 3), (2, 2)),
            UpSample(64, 32, (5, 5), (2, 2)),
            UpSample(32, 16, (7, 7), (2, 2)),
            UpSample(16, 8, (3, 3)),
            UpSample(8, 3, (4, 4), (1, 1), (2, 2))
        )

    def forward(self, x):
        return self.model(x)
