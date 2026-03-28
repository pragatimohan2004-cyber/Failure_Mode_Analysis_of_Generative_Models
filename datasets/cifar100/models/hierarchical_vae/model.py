import torch
import torch.nn as nn
import torch.nn.functional as F


class LadderVAE(nn.Module):
    def __init__(self, z_dims=[64, 32, 16]):
        super().__init__()

        self.z_dims = z_dims

        # ===== ENCODER =====
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 32 → 16
            nn.ReLU()
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 → 8
            nn.ReLU()
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),  # 8 → 4
            nn.ReLU()
        )

        # ===== LATENT PARAMS =====
        self.fc_mu1 = nn.Linear(64 * 16 * 16, z_dims[0])
        self.fc_logvar1 = nn.Linear(64 * 16 * 16, z_dims[0])

        self.fc_mu2 = nn.Linear(128 * 8 * 8, z_dims[1])
        self.fc_logvar2 = nn.Linear(128 * 8 * 8, z_dims[1])

        self.fc_mu3 = nn.Linear(256 * 4 * 4, z_dims[2])
        self.fc_logvar3 = nn.Linear(256 * 4 * 4, z_dims[2])

        # ===== DECODER =====
        self.fc_decode3 = nn.Linear(z_dims[2], 256 * 4 * 4)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 → 8
            nn.ReLU()
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8 → 16
            nn.ReLU()
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # 16 → 32
            nn.Tanh()
        )

        # ===== PROJECTION LAYERS (FIX) =====
        self.z2_proj = nn.Linear(z_dims[1], 128)
        self.z1_proj = nn.Linear(z_dims[0], 64)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        # ===== ENCODE =====
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)

        # ===== LATENTS =====
        mu1 = self.fc_mu1(h1.view(x.size(0), -1))
        logvar1 = self.fc_logvar1(h1.view(x.size(0), -1))

        mu2 = self.fc_mu2(h2.view(x.size(0), -1))
        logvar2 = self.fc_logvar2(h2.view(x.size(0), -1))

        mu3 = self.fc_mu3(h3.view(x.size(0), -1))
        logvar3 = self.fc_logvar3(h3.view(x.size(0), -1))

        z1 = self.reparameterize(mu1, logvar1)
        z2 = self.reparameterize(mu2, logvar2)
        z3 = self.reparameterize(mu3, logvar3)

        # ===== DECODE =====

        # Step 1: z3 → coarse features
        d3 = self.fc_decode3(z3).view(x.size(0), 256, 4, 4)

        # Step 2: inject z2 (mid-level)
        z2_proj = self.z2_proj(z2).unsqueeze(-1).unsqueeze(-1)  # [B,128,1,1]
        d2 = self.dec3(d3) + z2_proj  # broadcasting

        # Step 3: inject z1 (fine details)
        z1_proj = self.z1_proj(z1).unsqueeze(-1).unsqueeze(-1)  # [B,64,1,1]
        d1 = self.dec2(d2) + z1_proj

        # Step 4: reconstruct
        recon = self.dec1(d1)

        return recon, [mu1, mu2, mu3], [logvar1, logvar2, logvar3]