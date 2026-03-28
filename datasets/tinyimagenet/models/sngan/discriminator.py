import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()

        self.main = nn.Sequential(
            # 64 → 32
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 → 16
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 → 8
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 → 4
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 🔥 EXTRA FEATURE BLOCK (NEW)
            spectral_norm(nn.Conv2d(ndf*8, ndf*8, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 → 1
            spectral_norm(nn.Conv2d(ndf*8, 1, 4, 1, 0))
        )

    def forward(self, x):
        return self.main(x).view(-1)