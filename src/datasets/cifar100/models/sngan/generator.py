import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, nz=128, ngf=256, nc=3):
        super(Generator, self).__init__()

        self.main = nn.Sequential(

            # Input: Z → 4x4
            nn.ConvTranspose2d(nz, ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 4x4 → 8x8
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            # 8x8 → 16x16
            nn.ConvTranspose2d(ngf // 2, ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),

            # 16x16 → 32x32
            nn.ConvTranspose2d(ngf // 4, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)