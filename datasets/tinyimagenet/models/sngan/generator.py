import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()

        self.init = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
        )

        self.block1 = self._block(ngf*8, ngf*4)  # 4 → 8
        self.block2 = self._block(ngf*4, ngf*2)  # 8 → 16
        self.block3 = self._block(ngf*2, ngf)    # 16 → 32

        # 🔥 refinement block (important)
        self.refine = nn.Sequential(
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Final upsample to 64x64
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),   # 🔥 better than ConvTranspose
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Tanh()
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2),          # 🔥 key change
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.init(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.refine(x)
        x = self.final(x)
        return x