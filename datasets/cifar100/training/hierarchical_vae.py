import sys, os
sys.path.append(os.path.abspath(".."))

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import csv

from ladder_VAE_model import LadderVAE


# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== DIRECTORIES =====
os.makedirs("checkpoints_lvae", exist_ok=True)


# ===== LOSS STORAGE =====
loss_history = {
    "total_loss": [],
    "recon_loss": [],
    "kl_total": [],
    "kl_z1": [],
    "kl_z2": [],
    "kl_z3": []
}

csv_path = "checkpoints_lvae/loss_log.csv"

# Initialize CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "total_loss", "recon_loss", "kl_total", "kl_z1", "kl_z2", "kl_z3", "beta"])


# ===== DATA =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


# ===== MODEL =====
model = LadderVAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4)


# ===== LOSS FUNCTION =====
def ladder_loss(recon, x, mus, logvars, beta=1.0):

    recon_loss = F.mse_loss(recon, x, reduction='mean')

    kl_losses = []
    for mu, logvar in zip(mus, logvars):
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_losses.append(kl)

    kl_total = sum(kl_losses)
    total_loss = recon_loss + beta * kl_total

    return total_loss, recon_loss, kl_losses, kl_total


# ===== TRAIN =====
epochs = 100

for epoch in range(epochs):

    model.train()

    total_loss_epoch = 0
    total_recon_epoch = 0
    total_kl_epoch = 0

    kl_layer_sums = [0, 0, 0]

    beta = min(1.0, epoch / 10)

    for images, _ in dataloader:
        images = images.to(device)

        recon, mus, logvars = model(images)

        loss, recon_loss, kl_losses, kl_total = ladder_loss(
            recon, images, mus, logvars, beta
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()
        total_recon_epoch += recon_loss.item()
        total_kl_epoch += kl_total.item()

        for i in range(len(kl_losses)):
            kl_layer_sums[i] += kl_losses[i].item()

    # ===== AVERAGES =====
    avg_loss = total_loss_epoch / len(dataloader)
    avg_recon = total_recon_epoch / len(dataloader)
    avg_kl = total_kl_epoch / len(dataloader)
    avg_kl_layers = [k / len(dataloader) for k in kl_layer_sums]

    # ===== STORE =====
    loss_history["total_loss"].append(avg_loss)
    loss_history["recon_loss"].append(avg_recon)
    loss_history["kl_total"].append(avg_kl)
    loss_history["kl_z1"].append(avg_kl_layers[0])
    loss_history["kl_z2"].append(avg_kl_layers[1])
    loss_history["kl_z3"].append(avg_kl_layers[2])

    # ===== PRINT =====
    print(f"""
Epoch {epoch}
Total: {avg_loss:.4f}
Recon: {avg_recon:.4f}
KL Total: {avg_kl:.4f}
KL z1: {avg_kl_layers[0]:.4f}
KL z2: {avg_kl_layers[1]:.4f}
KL z3: {avg_kl_layers[2]:.4f}
Beta: {beta:.2f}
""")

    # ===== SAVE CSV =====
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            avg_loss,
            avg_recon,
            avg_kl,
            avg_kl_layers[0],
            avg_kl_layers[1],
            avg_kl_layers[2],
            beta
        ])

    # ===== SAVE JSON (periodic) =====
    if epoch % 10 == 0:
        with open("checkpoints_lvae/loss_log.json", "w") as f:
            json.dump(loss_history, f, indent=4)

    # ===== SAVE CHECKPOINT =====
    if epoch % 10 == 0:
        torch.save(
            model.state_dict(),
            f"checkpoints_lvae/lvae_epoch_{epoch}.pth"
        )


# ===== FINAL SAVE =====
torch.save(model.state_dict(), "checkpoints_lvae/lvae_final.pth")

with open("checkpoints_lvae/loss_log_final.json", "w") as f:
    json.dump(loss_history, f, indent=4)

print("✅ Ladder VAE Training Complete (model + detailed losses saved)")