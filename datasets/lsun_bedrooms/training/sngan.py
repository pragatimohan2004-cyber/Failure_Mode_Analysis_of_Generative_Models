import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import os
import sys
import json

sys.path.append("../")

from model.sngan_generator import Generator
from model.sngan_discriminator import Discriminator
from data.dataloader import get_dataloader


# ================= CONFIG =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 32
epochs = 100
nz = 100


# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

checkpoint_dir = os.path.join(BASE_DIR, "checkpoints_lsun")
generated_dir = os.path.join(BASE_DIR, "generated_lsun")

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)


# ================= DATA =================
dataloader = get_dataloader(
    root_dir="../data/lsun_clean",
    batch_size=batch_size,
    image_size=128
)


# ================= MODELS =================
netG = Generator(nz=nz).to(device)
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


# ================= OPTIMIZERS =================
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.0, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.0, 0.9))


# ================= FIXED NOISE =================
fixed_noise = torch.randn(64, nz, 1, 1, device=device)


# ================= LOSS STORAGE =================
G_losses = []
D_losses = []

epoch_G_losses = []
epoch_D_losses = []


print("\n🚀 Starting LSUN SNGAN Training...\n")


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
        # STORE LOSSES (iteration)
        # ============================
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        running_G.append(loss_G.item())
        running_D.append(loss_D.item())

        # ============================
        # Logging
        # ============================
        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"[Batch {i}] "
                f"Loss_D: {loss_D.item():.4f} "
                f"Loss_G: {loss_G.item():.4f}"
            )

    # ============================
    # STORE EPOCH LOSSES
    # ============================
    epoch_G_losses.append(sum(running_G) / len(running_G))
    epoch_D_losses.append(sum(running_D) / len(running_D))

    print(
        f"✅ Epoch {epoch} Avg -> "
        f"G: {epoch_G_losses[-1]:.4f}, "
        f"D: {epoch_D_losses[-1]:.4f}"
    )

    # ============================
    # SAVE SAMPLE IMAGES
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
torch.save(netG.state_dict(), f"{checkpoint_dir}/generator_final.pth")


# ================= SAVE LOSSES =================
with open(os.path.join(checkpoint_dir, "losses.json"), "w") as f:
    json.dump({
        "G_losses": G_losses,
        "D_losses": D_losses,
        "epoch_G_losses": epoch_G_losses,
        "epoch_D_losses": epoch_D_losses
    }, f)

print("Losses saved to checkpoints_lsun/losses.json")
print("Training Complete!")