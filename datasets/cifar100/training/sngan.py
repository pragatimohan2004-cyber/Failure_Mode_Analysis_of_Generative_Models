import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
import sys
import json

sys.path.append("../")

from dataloader_cifar100_DCGAN import get_dataloader
from sngan_generator_cifar100 import Generator
from sngan_discriminator_cifar100 import Discriminator


# ================= CONFIG =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 64
epochs = 100
nz = 128


# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

checkpoint_dir = os.path.join(BASE_DIR, "checkpoints_sngan")
generated_dir = os.path.join(BASE_DIR, "generated_sngan")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)


# ================= DATA =================
dataloader = get_dataloader(
    data_root="./processed/train",
    batch_size=batch_size,
    image_size=32
)


# ================= MODELS =================
netG = Generator(nz=nz).to(device)
netD = Discriminator().to(device)


# ================= OPTIMIZERS =================
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.0, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.0, 0.9))


# ================= FIXED NOISE =================
fixed_noise = torch.randn(64, nz, 1, 1, device=device)


# ================= LOSS TRACKING =================
G_losses = []
D_losses = []

epoch_G_losses = []
epoch_D_losses = []


print("\n🚀 Starting CIFAR100 SNGAN Training...\n")


# ================= TRAINING LOOP =================
for epoch in range(epochs):

    running_G = []
    running_D = []

    for i, (images, _) in enumerate(dataloader):

        real = images.to(device)
        b_size = real.size(0)

        # ============================
        # Train Discriminator (HINGE)
        # ============================
        netD.zero_grad()

        output_real = netD(real)
        loss_real = torch.mean(torch.relu(1.0 - output_real))

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)

        output_fake = netD(fake.detach())
        loss_fake = torch.mean(torch.relu(1.0 + output_fake))

        loss_D = loss_real + loss_fake
        loss_D.backward()

        # 🔥 stability (optional but recommended)
        torch.nn.utils.clip_grad_norm_(netD.parameters(), 5.0)

        optimizerD.step()

        # ============================
        # Train Generator (HINGE)
        # ============================
        netG.zero_grad()

        output = netD(fake)
        loss_G = -torch.mean(output)

        loss_G.backward()

        torch.nn.utils.clip_grad_norm_(netG.parameters(), 5.0)

        optimizerG.step()

        # ============================
        # STORE LOSSES
        # ============================
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        running_G.append(loss_G.item())
        running_D.append(loss_D.item())

        # ============================
        # LOGGING
        # ============================
        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"[Batch {i}] "
                f"Loss_D: {loss_D.item():.4f} "
                f"Loss_G: {loss_G.item():.4f}"
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
    # SAVE LOSSES (EVERY EPOCH 🔥)
    # ============================
    with open(os.path.join(checkpoint_dir, "losses.json"), "w") as f:
        json.dump({
            "G_losses": G_losses,
            "D_losses": D_losses,
            "epoch_G_losses": epoch_G_losses,
            "epoch_D_losses": epoch_D_losses
        }, f)

    print("📊 Losses updated")

    # ============================
    # SAVE GENERATED IMAGES
    # ============================
    if epoch % 5 == 0:
        with torch.no_grad():
            fake_images = netG(fixed_noise).cpu()

            vutils.save_image(
                fake_images,
                f"{generated_dir}/epoch_{epoch}.png",
                normalize=True,
                nrow=8
            )

    # ============================
    # SAVE CHECKPOINTS
    # ============================
    if epoch % 10 == 0:
        torch.save({
            "epoch": epoch,
            "generator": netG.state_dict(),
            "discriminator": netD.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "optimizerD": optimizerD.state_dict()
        }, f"{checkpoint_dir}/checkpoint_{epoch}.pth")

        print(f"💾 Saved checkpoint at epoch {epoch}")


# ================= FINAL SAVE =================
torch.save(netG.state_dict(), f"{checkpoint_dir}/sngan_cifar_generator_final.pth")

print("\n Training Complete!")