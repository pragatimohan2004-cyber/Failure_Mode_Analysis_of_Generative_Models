import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # 128 → 64
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 → 32
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 → 16
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 → 8
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 → 4
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 → 1
            spectral_norm(nn.Conv2d(ndf * 16, 1, 4, 1, 0))
        )

    def forward(self, x):
        return self.main(x).view(-1)