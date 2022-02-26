import torch.nn as nn
import torch
from block import DownSample


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            DownSample(in_channels, 8, (3, 3)),
            DownSample(8, 16, (3, 3)),
            DownSample(16, 32, (3, 3), (2, 2)),
            DownSample(32, 64, (3, 3)),
            DownSample(64, 128, (5, 5), (2, 2)),
            DownSample(128, 128, (3, 3)),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)
