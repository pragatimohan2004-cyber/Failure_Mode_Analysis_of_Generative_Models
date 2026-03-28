import torch
import torch.nn as nn
import torch.nn.functional as F


class LadderVAE(nn.Module):
    def __init__(self, z_dims=[64, 128, 256]):
        super().__init__()

        self.z_dims = z_dims

        # ================= ENCODER (BOTTOM-UP) =================
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1)     # 64 → 32
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)   # 32 → 16
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)  # 16 → 8
        self.enc4 = nn.Conv2d(256, 512, 4, 2, 1)  # 8 → 4

        # Posterior (bottom-up)
        self.qz3_mu = nn.Linear(512 * 4 * 4, z_dims[2])
        self.qz3_logvar = nn.Linear(512 * 4 * 4, z_dims[2])

        self.qz2_mu = nn.Linear(256 * 8 * 8, z_dims[1])
        self.qz2_logvar = nn.Linear(256 * 8 * 8, z_dims[1])

        self.qz1_mu = nn.Linear(128 * 16 * 16, z_dims[0])
        self.qz1_logvar = nn.Linear(128 * 16 * 16, z_dims[0])

        # ================= DECODER (TOP-DOWN) =================
        self.fc_z3 = nn.Linear(z_dims[2], 512 * 4 * 4)

        self.dec3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)  # 4 → 8
        self.dec2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 8 → 16
        self.dec1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 16 → 32
        self.dec0 = nn.ConvTranspose2d(64, 3, 4, 2, 1)     # 32 → 64

        # Prior networks (top-down)
        self.pz2_mu = nn.Linear(z_dims[2], z_dims[1])
        self.pz2_logvar = nn.Linear(z_dims[2], z_dims[1])

        self.pz1_mu = nn.Linear(z_dims[1], z_dims[0])
        self.pz1_logvar = nn.Linear(z_dims[1], z_dims[0])


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    # ================= FORWARD =================
    def forward(self, x):

        # -------- Bottom-up --------
        h1 = F.relu(self.enc1(x))   # 32×32
        h2 = F.relu(self.enc2(h1))  # 16×16
        h3 = F.relu(self.enc3(h2))  # 8×8
        h4 = F.relu(self.enc4(h3))  # 4×4

        # Flatten
        h4_flat = h4.view(h4.size(0), -1)
        h3_flat = h3.view(h3.size(0), -1)
        h2_flat = h2.view(h2.size(0), -1)

        # Posterior q(z|x)
        mu3, logvar3 = self.qz3_mu(h4_flat), self.qz3_logvar(h4_flat)
        z3 = self.reparameterize(mu3, logvar3)

        mu2_q, logvar2_q = self.qz2_mu(h3_flat), self.qz2_logvar(h3_flat)
        mu1_q, logvar1_q = self.qz1_mu(h2_flat), self.qz1_logvar(h2_flat)

        # -------- Top-down --------
        # z3 → z2 prior
        mu2_p, logvar2_p = self.pz2_mu(z3), self.pz2_logvar(z3)

        # Combine (Ladder trick)
        mu2 = mu2_q + mu2_p
        logvar2 = logvar2_q + logvar2_p
        z2 = self.reparameterize(mu2, logvar2)

        # z2 → z1 prior
        mu1_p, logvar1_p = self.pz1_mu(z2), self.pz1_logvar(z2)

        mu1 = mu1_q + mu1_p
        logvar1 = logvar1_q + logvar1_p
        z1 = self.reparameterize(mu1, logvar1)

        # -------- Decoder --------
        h = self.fc_z3(z3)
        h = h.view(-1, 512, 4, 4)

        h = F.relu(self.dec3(h))
        h = F.relu(self.dec2(h))
        h = F.relu(self.dec1(h))
        x_recon = torch.sigmoid(self.dec0(h))

        return x_recon, [(mu1, logvar1), (mu2, logvar2), (mu3, logvar3)]