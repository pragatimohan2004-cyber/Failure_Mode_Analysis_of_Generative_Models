import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):

    def __init__(self, nc=3, ndf=128):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(

            # 32x32 → 16x16
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

            # 16x16 → 8x8
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

            # 8x8 → 4x4
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),

            # 4x4 → 1x1
            spectral_norm(nn.Conv2d(ndf * 4, 1, 4, 1, 0))
        )

    def forward(self, x):
        return self.main(x).view(-1)