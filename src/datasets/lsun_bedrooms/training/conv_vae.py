import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import json
import csv

from data.dataloader import get_dataloader
from model.convolutional_vae_model import ConvVAE_LSUN


# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== DIRECTORIES =====
os.makedirs("checkpoints_vae", exist_ok=True)
os.makedirs("vae_samples", exist_ok=True)


# ===== LOSS STORAGE =====
loss_history = {
    "total_loss": [],
    "recon_loss": [],
    "kl_loss": []
}

csv_path = "checkpoints_vae/loss_log.csv"

# Initialize CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "total_loss", "recon_loss", "kl_loss", "beta"])


# ===== DATA =====
dataloader = get_dataloader(
    root_dir="../data/lsun_clean",
    batch_size=32,
    image_size=128
)


# ===== MODEL =====
model = ConvVAE_LSUN().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)


# ===== LOSS FUNCTION =====
def vae_loss(recon, x, mu, logvar, beta):
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl
    return total, recon_loss, kl


# ===== TRAIN =====
epochs = 100

for epoch in range(epochs):

    model.train()

    total_loss = 0
    total_recon = 0
    total_kl = 0

    beta = min(1.0, epoch / 10)  # KL annealing

    for batch in dataloader:

        # Handle dataset output
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device)

        recon, mu, logvar = model(images)

        loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    # ===== AVERAGE LOSSES =====
    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon / len(dataloader)
    avg_kl = total_kl / len(dataloader)

    # ===== STORE LOSSES =====
    loss_history["total_loss"].append(avg_loss)
    loss_history["recon_loss"].append(avg_recon)
    loss_history["kl_loss"].append(avg_kl)

    # ===== PRINT =====
    print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Beta: {beta:.2f}")

    # ===== SAVE CSV =====
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_loss, avg_recon, avg_kl, beta])

    # ===== SAVE JSON (periodic safety) =====
    if epoch % 10 == 0:
        with open("checkpoints_vae/loss_log.json", "w") as f:
            json.dump(loss_history, f, indent=4)

    # ===== SAVE RECONSTRUCTIONS =====
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            sample = images[:16]
            recon_sample, _, _ = model(sample)

            # Denormalize [-1,1] → [0,1]
            comparison = torch.cat([
                (sample + 1) / 2,
                (recon_sample + 1) / 2
            ])

            vutils.save_image(
                comparison,
                f"vae_samples/recon_epoch_{epoch}.png",
                nrow=8
            )

    # ===== SAVE CHECKPOINT =====
    if epoch % 10 == 0:
        torch.save(
            model.state_dict(),
            f"checkpoints_vae/vae_epoch_{epoch}.pth"
        )


# ===== FINAL SAVE =====
torch.save(model.state_dict(), "checkpoints_vae/vae_lsun_final.pth")

with open("checkpoints_vae/loss_log_final.json", "w") as f:
    json.dump(loss_history, f, indent=4)

print("\n✅ LSUN VAE Training Complete (model + losses saved)")