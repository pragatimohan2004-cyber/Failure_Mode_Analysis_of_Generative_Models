import torch
import torch.nn as nn
import torch.nn.functional as F


class LadderVAE(nn.Module):
    def __init__(self, z_dims=[64, 128, 256]):
        super().__init__()

        # ================= ENCODER =================
        self.enc1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.enc4 = nn.Conv2d(256, 512, 4, 2, 1)

        # 🔥 ADD NORMALIZATION (CRITICAL)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # ================= LATENT =================
        self.qz3_mu = nn.Linear(512 * 4 * 4, z_dims[2])
        self.qz3_logvar = nn.Linear(512 * 4 * 4, z_dims[2])

        self.qz2_mu = nn.Linear(256 * 8 * 8, z_dims[1])
        self.qz2_logvar = nn.Linear(256 * 8 * 8, z_dims[1])

        self.qz1_mu = nn.Linear(128 * 16 * 16, z_dims[0])
        self.qz1_logvar = nn.Linear(128 * 16 * 16, z_dims[0])

        # ================= DECODER =================
        self.fc_z3 = nn.Linear(z_dims[2], 512 * 4 * 4)

        self.dec3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec0 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

        # 🔥 DECODER NORMALIZATION
        self.dbn3 = nn.BatchNorm2d(256)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn1 = nn.BatchNorm2d(64)

        # ================= PRIORS =================
        self.pz2_mu = nn.Linear(z_dims[2], z_dims[1])
        self.pz2_logvar = nn.Linear(z_dims[2], z_dims[1])

        self.pz1_mu = nn.Linear(z_dims[1], z_dims[0])
        self.pz1_logvar = nn.Linear(z_dims[1], z_dims[0])


    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -10, 10)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std


    def fuse_gaussians(self, mu_q, logvar_q, mu_p, logvar_p):

        logvar_q = torch.clamp(logvar_q, -10, 10)
        logvar_p = torch.clamp(logvar_p, -10, 10)

        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)

        precision_q = 1.0 / (var_q + 1e-6)
        precision_p = 1.0 / (var_p + 1e-6)

        var = 1.0 / (precision_q + precision_p + 1e-6)

        mu = var * (mu_q * precision_q + mu_p * precision_p)
        logvar = torch.log(var + 1e-6)

        return mu, torch.clamp(logvar, -10, 10)


    def forward(self, x):

        # -------- ENCODER --------
        h1 = F.leaky_relu(self.bn1(self.enc1(x)), 0.2)
        h2 = F.leaky_relu(self.bn2(self.enc2(h1)), 0.2)
        h3 = F.leaky_relu(self.bn3(self.enc3(h2)), 0.2)
        h4 = F.leaky_relu(self.bn4(self.enc4(h3)), 0.2)

        h4_flat = h4.view(h4.size(0), -1)
        h3_flat = h3.view(h3.size(0), -1)
        h2_flat = h2.view(h2.size(0), -1)

        # -------- POSTERIOR --------
        mu3 = self.qz3_mu(h4_flat)
        logvar3 = torch.clamp(self.qz3_logvar(h4_flat), -10, 10)
        z3 = self.reparameterize(mu3, logvar3)

        mu2_q = self.qz2_mu(h3_flat)
        logvar2_q = torch.clamp(self.qz2_logvar(h3_flat), -10, 10)

        mu1_q = self.qz1_mu(h2_flat)
        logvar1_q = torch.clamp(self.qz1_logvar(h2_flat), -10, 10)

        # -------- TOP-DOWN --------
        mu2_p = self.pz2_mu(z3)
        logvar2_p = torch.clamp(self.pz2_logvar(z3), -10, 10)

        mu2, logvar2 = self.fuse_gaussians(mu2_q, logvar2_q, mu2_p, logvar2_p)
        z2 = self.reparameterize(mu2, logvar2)

        mu1_p = self.pz1_mu(z2)
        logvar1_p = torch.clamp(self.pz1_logvar(z2), -10, 10)

        mu1, logvar1 = self.fuse_gaussians(mu1_q, logvar1_q, mu1_p, logvar1_p)
        z1 = self.reparameterize(mu1, logvar1)

        # -------- DECODER --------
        h = self.fc_z3(z3)
        h = h.view(-1, 512, 4, 4)

        h = F.leaky_relu(self.dbn3(self.dec3(h)), 0.2)
        h = F.leaky_relu(self.dbn2(self.dec2(h)), 0.2)
        h = F.leaky_relu(self.dbn1(self.dec1(h)), 0.2)

        # 🔥 CRITICAL FIX
        x_recon = torch.tanh(self.dec0(h))

        return x_recon, [(mu1, logvar1), (mu2, logvar2), (mu3, logvar3)]