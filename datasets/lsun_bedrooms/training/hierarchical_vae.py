import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import json

# ================= PATH FIX =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from model.ladder_vae_model import LadderVAE
from data.dataloader import LSUNBedroomDataset
from torch.utils.data import DataLoader


# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================= CONFIG =================
batch_size = 32
epochs = 50
lr = 1e-4


# ================= DATA =================
dataset = LSUNBedroomDataset("../data/lsun_clean")

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


# ================= MODEL =================
model = LadderVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


# ================= LOSS FUNCTION =================
def ladder_vae_loss(x_recon, x, latents, beta=1.0):

    # ✅ FIX 1: normalized reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction="mean")

    # ✅ FIX 2: normalized KL
    kl_loss = 0
    for mu, logvar in latents:
        kl_loss += -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ================= LOSS TRACKING =================
total_losses = []
recon_losses = []
kl_losses = []


print("\n🚀 Starting LSUN Ladder VAE Training...\n")


# ================= TRAINING LOOP =================
for epoch in range(epochs):

    model.train()
    running_loss = 0

    # ✅ KL annealing
    beta = min(1.0, epoch / 10)

    for batch in dataloader:

        # ✅ FIX 3: handle dataloader formats safely
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(device)

        # Forward
        recon, latents = model(images)

        loss, recon_l, kl_l = ladder_vae_loss(recon, images, latents, beta)

        optimizer.zero_grad()
        loss.backward()

        # ✅ FIX 4: gradient clipping (CRITICAL for VAE stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        # Store losses
        total_losses.append(loss.item())
        recon_losses.append(recon_l.item())
        kl_losses.append(kl_l.item())

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)

    print(
        f"Epoch {epoch:03d} | "
        f"Loss: {avg_loss:.4f} | "
        f"Recon: {recon_l:.4f} | "
        f"KL: {kl_l:.4f} | "
        f"Beta: {beta:.2f}"
    )


# ================= SAVE =================
save_dir = "../checkpoints_lvae"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), f"{save_dir}/ladder_vae_lsun.pth")

with open(f"{save_dir}/losses.json", "w") as f:
    json.dump({
        "total_loss": total_losses,
        "recon_loss": recon_losses,
        "kl_loss": kl_losses
    }, f)

print("\n📊 Losses saved to checkpoints_lvae/")
print("✅ Training Complete!")