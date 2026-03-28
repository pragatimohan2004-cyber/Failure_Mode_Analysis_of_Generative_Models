import sys, os
sys.path.append(os.path.abspath(".."))

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity as ssim

from ladder_VAE_model import LadderVAE


# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== DATA =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

dataset = datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


# ===== MODEL =====
model = LadderVAE().to(device)

checkpoint_path = "checkpoints_lvae/lvae_final.pth"
assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

print("✅ Ladder VAE Loaded")


# ===== METRICS =====
total_mse = 0
total_ssim = 0
count = 0


with torch.no_grad():
    for images, _ in dataloader:

        images = images.to(device)

        recon, _, _ = model(images)

        # ===== MSE =====
        mse = F.mse_loss(recon, images, reduction='mean')
        total_mse += mse.item()

        # ===== SSIM =====
        recon_np = (recon.cpu().numpy() + 1) / 2   # [-1,1] → [0,1]
        images_np = (images.cpu().numpy() + 1) / 2

        for i in range(images.size(0)):
            total_ssim += ssim(
                images_np[i].transpose(1, 2, 0),
                recon_np[i].transpose(1, 2, 0),
                channel_axis=2,
                data_range=1
            )
            count += 1


# ===== RESULTS =====
avg_mse = total_mse / len(dataloader)
avg_ssim = total_ssim / count

print("\n📊 Ladder VAE Evaluation Results")
print(f"MSE  : {avg_mse:.6f}")
print(f"SSIM : {avg_ssim:.4f}")