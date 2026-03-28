import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import json

# ================= PATH FIX =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from ladder_vae_model import LadderVAE
from dataloader_tinyimagenet import get_dataloader


# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================= CONFIG =================
batch_size = 64
epochs = 100
lr = 5e-5   # 🔥 LOWER LR


# ================= DATA =================
dataloader = get_dataloader(
    root_dir="tiny-imagenet-200/train",
    batch_size=batch_size
)


# ================= MODEL =================
model = LadderVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


# ================= LOSS FUNCTION =================
def ladder_loss(x_recon, x, latents, beta):

    # ✅ Reconstruction (stable)
    recon = F.mse_loss(x_recon, x, reduction="mean")

    # ✅ Multi-level KL (normalized)
    kl = 0
    for mu, logvar in latents:
        kl += -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    kl = kl / len(latents)   # 🔥 CRITICAL FIX

    return recon + beta * kl, recon, kl


# ================= LOSS TRACKING =================
total_losses = []
recon_losses = []
kl_losses = []


print("\n🚀 Training TinyImageNet Ladder VAE...\n")


# ================= TRAINING LOOP =================
for epoch in range(epochs):

    model.train()
    running = 0

    # 🔥 SLOW KL ANNEALING
    beta = min(1.0, epoch / 20)

    for i, batch in enumerate(dataloader):

        # Handle dataset output
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(device)

        # Forward
        recon, latents = model(images)

        # 🔥 DEBUG NaN CHECK
        if torch.isnan(recon).any():
            print("❌ NaN in reconstruction — stopping")
            break

        loss, recon_l, kl_l = ladder_loss(recon, images, latents, beta)

        optimizer.zero_grad()
        loss.backward()

        # 🔥 GRADIENT CLIPPING
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        # Store
        total_losses.append(loss.item())
        recon_losses.append(recon_l.item())
        kl_losses.append(kl_l.item())

        running += loss.item()

        if i % 100 == 0:
            print(
                f"Epoch {epoch} | Batch {i} | "
                f"Loss: {loss.item():.4f} | "
                f"Recon: {recon_l.item():.4f} | "
                f"KL: {kl_l.item():.4f}"
            )

    avg_loss = running / len(dataloader)

    print(f"✅ Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f} | Beta: {beta:.2f}")


# ================= SAVE =================
save_dir = os.path.join(BASE_DIR, "checkpoints_lvae")
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(save_dir, "ladder_vae_tiny.pth"))

with open(os.path.join(save_dir, "losses.json"), "w") as f:
    json.dump({
        "total_loss": total_losses,
        "recon_loss": recon_losses,
        "kl_loss": kl_losses
    }, f)

print("\n📊 Losses saved")  
print("✅ Training Complete!")