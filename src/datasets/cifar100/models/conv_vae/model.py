import torch
import torch.nn as nn


class ConvVAE(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()

        # ===== ENCODER =====
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 32 → 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16 → 8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 8 → 4
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, z_dim)

        # ===== DECODER =====
        self.fc_decode = nn.Linear(z_dim, 128 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        enc = enc.view(x.size(0), -1)

        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)

        z = self.reparameterize(mu, logvar)

        dec = self.fc_decode(z)
        dec = dec.view(x.size(0), 128, 4, 4)

        recon = self.decoder(dec)

        return recon, mu, logvar