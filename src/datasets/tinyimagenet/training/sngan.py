from dataloader_tinyimagenet import get_dataloader
from sngan_generator import Generator
from sngan_discriminator import Discriminator

import torch
import torch.optim as optim
import torchvision.utils as vutils
import os
import json


# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================= CONFIG =================
batch_size = 64
epochs = 100
nz = 100


# ================= DATA =================
dataloader = get_dataloader("tiny-imagenet-200/train", batch_size)


# ================= MODELS =================
netG = Generator(nz).to(device)
netD = Discriminator().to(device)


# ================= OPTIMIZERS =================
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.0, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.0, 0.9))


# ================= FIXED NOISE =================
fixed_noise = torch.randn(64, nz, 1, 1, device=device)


# ================= DIRECTORIES =================
os.makedirs("generated_tiny", exist_ok=True)
os.makedirs("checkpoints_tiny", exist_ok=True)


# ================= LOSS TRACKING =================
G_losses = []
D_losses = []

epoch_G_losses = []
epoch_D_losses = []


print("\n🚀 Starting TinyImageNet SNGAN Training...\n")


# ================= TRAINING LOOP =================
for epoch in range(epochs):

    running_G = []
    running_D = []

    for i, images in enumerate(dataloader):

        real = images.to(device)
        b_size = real.size(0)

        # ============================
        # Train Discriminator
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
        optimizerD.step()

        # ============================
        # Train Generator
        # ============================
        netG.zero_grad()

        output = netD(fake)
        loss_G = -torch.mean(output)

        loss_G.backward()
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
    # EPOCH-LEVEL LOSSES
    # ============================
    epoch_G = sum(running_G) / len(running_G)
    epoch_D = sum(running_D) / len(running_D)

    epoch_G_losses.append(epoch_G)
    epoch_D_losses.append(epoch_D)

    print(f"✅ Epoch {epoch} Avg → G: {epoch_G:.4f}, D: {epoch_D:.4f}")

    # ============================
    # SAVE GENERATED IMAGES
    # ============================
    if epoch % 5 == 0:
        with torch.no_grad():
            fake_images = netG(fixed_noise).cpu()

            vutils.save_image(
                fake_images,
                f"generated_tiny/epoch_{epoch}.png",
                normalize=True,  # OK for visualization
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
        }, f"checkpoints_tiny/checkpoint_{epoch}.pth")

        print(f"💾 Saved checkpoint at epoch {epoch}")


# ================= FINAL SAVE =================
torch.save(netG.state_dict(), "checkpoints_tiny/sngan_tiny_generator_final.pth")


# ================= SAVE LOSSES =================
with open("checkpoints_tiny/losses.json", "w") as f:
    json.dump({
        "G_losses": G_losses,
        "D_losses": D_losses,
        "epoch_G_losses": epoch_G_losses,
        "epoch_D_losses": epoch_D_losses
    }, f)

print("\n📊 Losses saved to checkpoints_tiny/losses.json")
print(" Training Complete!")