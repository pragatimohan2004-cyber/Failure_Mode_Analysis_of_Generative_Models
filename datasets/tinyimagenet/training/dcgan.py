import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import json

# ================= PATH FIX =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from dataloader_tinyimagenet import get_dataloader
from dcgan_generator import Generator, weights_init
from dcgan_discriminator import Discriminator

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 64
epochs = 100
nz = 100


# ================= PATH =================
checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)


# ================= DATA (🔥 FIXED) =================
dataloader = get_dataloader(
    root_dir="tiny-imagenet-200/train",   # ✅ FIXED
    batch_size=batch_size
)


# ================= MODELS =================
netG = Generator(nz=nz).to(device)
netD = Discriminator().to(device)

netG.apply(weights_init)
netD.apply(weights_init)


# ================= LOSS =================
criterion = nn.BCELoss()

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))


# ================= LOSS TRACKING =================
G_losses = []
D_losses = []

epoch_G_losses = []
epoch_D_losses = []


print("\n🚀 Starting TinyImageNet DCGAN Training...\n")


# ================= TRAINING LOOP =================
for epoch in range(epochs):

    running_G = []
    running_D = []

    for i, images in enumerate(dataloader):

        real = images.to(device)
        b_size = real.size(0)

        # label smoothing
        label_real = torch.full((b_size,), 0.9, device=device)
        label_fake = torch.zeros(b_size, device=device)

        # -----------------------
        # Train Discriminator
        # -----------------------
        netD.zero_grad()

        output_real = netD(real)
        loss_real = criterion(output_real, label_real)

        noise = torch.randn(b_size, nz, 1, 1, device=device)
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
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"[Batch {i}] "
                f"Loss_D: {loss_D.item():.4f} "
                f"Loss_G: {loss_G.item():.4f}"
            )

    # -----------------------
    # EPOCH LOSSES
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

        print(f"💾 Saved checkpoint at epoch {epoch}")


# -----------------------
# FINAL SAVE
# -----------------------
torch.save(netG.state_dict(), f"{checkpoint_dir}/dcgan_tinyimagenet_generator_final.pth")


# -----------------------
# SAVE LOSSES
# -----------------------
with open(os.path.join(checkpoint_dir, "losses.json"), "w") as f:
    json.dump({
        "G_losses": G_losses,
        "D_losses": D_losses,
        "epoch_G_losses": epoch_G_losses,
        "epoch_D_losses": epoch_D_losses
    }, f)

print("\n📊 Losses saved to checkpoints/losses.json")
print(" Training Complete!")