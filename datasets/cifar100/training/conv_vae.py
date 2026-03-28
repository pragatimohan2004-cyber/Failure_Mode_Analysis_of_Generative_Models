import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import json
import csv

from convolutional_VAE_model import ConvVAE


# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===== DIRECTORIES =====
os.makedirs("checkpoints_vae", exist_ok=True)


# ===== LOSS STORAGE =====
loss_history = {
    "total_loss": [],
    "recon_loss": [],
    "kl_loss": []
}

csv_path = "checkpoints_vae/loss_log.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "total_loss", "recon_loss", "kl_loss"])


# ===== DATA =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

dataset = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)

print("Dataset loaded:", len(dataset))


# ===== MODEL =====
model = ConvVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ===== LOSS =====
def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x, reduction='mean')  # ✅ FIXED
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # ✅ FIXED
    return recon_loss + kl, recon_loss, kl


# ===== TRAIN =====
epochs = 100

for epoch in range(epochs):

    model.train()

    total_loss = 0
    total_recon = 0
    total_kl = 0

    for images, _ in dataloader:
        images = images.to(device)

        recon, mu, logvar = model(images)

        loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    # ===== AVERAGES =====
    avg_loss = total_loss / len(dataloader)
    avg_recon = total_recon / len(dataloader)
    avg_kl = total_kl / len(dataloader)

    # ===== STORE =====
    loss_history["total_loss"].append(avg_loss)
    loss_history["recon_loss"].append(avg_recon)
    loss_history["kl_loss"].append(avg_kl)

    # ===== PRINT =====
    print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

    # ===== SAVE CSV =====
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_loss, avg_recon, avg_kl])

    # ===== SAVE JSON (periodic) =====
    if epoch % 10 == 0:
        with open("checkpoints_vae/loss_log.json", "w") as f:
            json.dump(loss_history, f, indent=4)

    # ===== SAVE CHECKPOINT =====
    if epoch % 10 == 0:
        torch.save(
            model.state_dict(),
            f"checkpoints_vae/vae_epoch_{epoch}.pth"
        )


# ===== FINAL SAVE =====
torch.save(model.state_dict(), "checkpoints_vae/vae_final.pth")

with open("checkpoints_vae/loss_log_final.json", "w") as f:
    json.dump(loss_history, f, indent=4)

print("✅ Training Complete (model + losses saved)")