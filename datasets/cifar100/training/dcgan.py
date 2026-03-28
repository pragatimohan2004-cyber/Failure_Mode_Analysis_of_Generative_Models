import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
import sys
import json

# allow import
sys.path.append("../")

from dataloader_cifar100_DCGAN import get_dataloader
from dcgan_generator import Generator
from dcgan_discriminator import Discriminator

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================= CONFIG =================
batch_size = 128
z_dim = 100
epochs = 100
lr = 0.0002
beta1 = 0.5


# ================= DATA =================
dataloader = get_dataloader(
    data_root="./processed/train",
    batch_size=batch_size,
    image_size=32
)


# ================= MODELS =================
G = Generator(z_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()

optG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))


# ================= PATHS =================
os.makedirs("generated_cifar", exist_ok=True)
os.makedirs("checkpoints_cifar", exist_ok=True)


# ================= FIXED NOISE =================
fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)


# ================= LOSS TRACKING =================
G_losses = []
D_losses = []

epoch_G_losses = []
epoch_D_losses = []


print("\n🚀 Starting DCGAN CIFAR100 Training...\n")


# ================= TRAINING LOOP =================
for epoch in range(epochs):

    running_G = []
    running_D = []

    for i, (real, _) in enumerate(dataloader):

        real = real.to(device)
        batch = real.size(0)

        real_label = torch.ones(batch, device=device)
        fake_label = torch.zeros(batch, device=device)


        # ============================
        # Train Discriminator
        # ============================
        noise = torch.randn(batch, z_dim, 1, 1, device=device)
        fake = G(noise)

        loss_real = criterion(D(real), real_label)
        loss_fake = criterion(D(fake.detach()), fake_label)

        lossD = loss_real + loss_fake

        optD.zero_grad()
        lossD.backward()
        optD.step()


        # ============================
        # Train Generator
        # ============================
        noise = torch.randn(batch, z_dim, 1, 1, device=device)
        fake = G(noise)

        lossG = criterion(D(fake), real_label)

        optG.zero_grad()
        lossG.backward()
        optG.step()


        # ============================
        # STORE LOSSES
        # ============================
        G_losses.append(lossG.item())
        D_losses.append(lossD.item())

        running_G.append(lossG.item())
        running_D.append(lossD.item())


        # ============================
        # LOGGING
        # ============================
        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"[Batch {i}] "
                f"Loss_D: {lossD.item():.4f} "
                f"Loss_G: {lossG.item():.4f}"
            )


    # ============================
    # EPOCH LOSSES
    # ============================
    epoch_G = sum(running_G) / len(running_G)
    epoch_D = sum(running_D) / len(running_D)

    epoch_G_losses.append(epoch_G)
    epoch_D_losses.append(epoch_D)

    print(f"✅ Epoch {epoch} Avg → G: {epoch_G:.4f}, D: {epoch_D:.4f}")


    # ============================
    # SAVE IMAGES
    # ============================
    if epoch % 5 == 0:
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()

        vutils.save_image(
            fake,
            f"generated_cifar/epoch_{epoch}.png",
            normalize=True,
            nrow=8
        )


    # ============================
    # SAVE CHECKPOINTS
    # ============================
    if epoch % 10 == 0:
        torch.save({
            "epoch": epoch,
            "generator": G.state_dict(),
            "discriminator": D.state_dict(),
            "optG": optG.state_dict(),
            "optD": optD.state_dict()
        }, f"checkpoints_cifar/checkpoint_{epoch}.pth")

        print(f"💾 Saved checkpoint at epoch {epoch}")


# ================= FINAL SAVE =================
torch.save(G.state_dict(), "checkpoints_cifar/dcgan_cifar_generator_final.pth")


# ================= SAVE LOSSES =================
with open("checkpoints_cifar/losses.json", "w") as f:
    json.dump({
        "G_losses": G_losses,
        "D_losses": D_losses,
        "epoch_G_losses": epoch_G_losses,
        "epoch_D_losses": epoch_D_losses
    }, f)

print("\n📊 Losses saved to checkpoints_cifar/losses.json")
print(" Training Complete!")