import sys
import os
import json

sys.path.append(os.path.abspath(".."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 

from model.generator import Generator
from model.discriminator import Discriminator
from data.dataloader import LSUNBedroomDataset


# ================= CONFIG =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64
epochs = 100
z_dim = 100

checkpoint_dir = "../checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)


# ================= DATA =================
dataset = LSUNBedroomDataset("../data/lsun_clean") 
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ================= MODELS =================
netG = Generator(z_dim).to(device)
netD = Discriminator().to(device)


# ================= WEIGHT INIT =================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

netG.apply(weights_init)
netD.apply(weights_init)


# ================= LOSS + OPTIM =================
criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

print("Using device:", device)


# ================= LOSS TRACKING =================
G_losses = []
D_losses = []

epoch_G_losses = []
epoch_D_losses = []


# ================= TRAINING LOOP =================
for epoch in range(epochs):

    running_G = []
    running_D = []

    for i, images in enumerate(dataloader):

        real = images.to(device)
        b_size = real.size(0)

        label_real = torch.ones(b_size, device=device)
        label_fake = torch.zeros(b_size, device=device)

        # -----------------------
        # Train Discriminator
        # -----------------------
        netD.zero_grad()

        output_real = netD(real)
        loss_real = criterion(output_real, label_real)

        noise = torch.randn(b_size, z_dim, 1, 1, device=device)
        fake = netG(noise)

        output_fake = netD(fake.detach())
        loss_fake = criterion(output_fake, label_fake)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # -----------------------
        # Train Generator
        # -----------------------
        netG.zero_grad()

        output = netD(fake)
        loss_G = criterion(output, label_real)

        loss_G.backward()
        optimizerG.step()

        # -----------------------
        # STORE LOSSES
        # -----------------------
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        running_G.append(loss_G.item())
        running_D.append(loss_D.item())

        # -----------------------
        # LOGGING
        # -----------------------
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch [{i}] Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}")

    # -----------------------
    # EPOCH-LEVEL LOSSES
    # -----------------------
    epoch_G = sum(running_G) / len(running_G)
    epoch_D = sum(running_D) / len(running_D)

    epoch_G_losses.append(epoch_G)
    epoch_D_losses.append(epoch_D)

    print(f"✅ Epoch {epoch} Avg → G: {epoch_G:.4f}, D: {epoch_D:.4f}")

    # -----------------------
    # SAVE CHECKPOINTS
    # -----------------------
    if epoch % 10 == 0:
        torch.save(netG.state_dict(), f"{checkpoint_dir}/generator_epoch_{epoch}.pth")
        torch.save(netD.state_dict(), f"{checkpoint_dir}/discriminator_epoch_{epoch}.pth")


# -----------------------
# FINAL SAVE
# -----------------------
torch.save(netG.state_dict(), f"{checkpoint_dir}/dcgan_lsun_generator_final.pth")


# -----------------------
# SAVE LOSSES (IMPORTANT)
# -----------------------
with open(f"{checkpoint_dir}/losses.json", "w") as f:
    json.dump({
        "G_losses": G_losses,
        "D_losses": D_losses,
        "epoch_G_losses": epoch_G_losses,
        "epoch_D_losses": epoch_D_losses
    }, f)

print("\n📊 Losses saved to ../checkpoints/losses.json")
print(" Training complete.")