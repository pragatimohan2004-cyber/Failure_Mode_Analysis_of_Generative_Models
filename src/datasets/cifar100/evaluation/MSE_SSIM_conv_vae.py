import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

from convolutional_VAE_model import ConvVAE


# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===== DATA =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3)
])

dataset = datasets.CIFAR100(
    root="./data",
    train=False,   # 🔥 use test set for evaluation
    download=True,
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


# ===== LOAD MODEL =====
model = ConvVAE().to(device)
model.load_state_dict(torch.load("./checkpoints_vae/vae_final.pth", map_location=device))
model.eval()

print("Model loaded!")


# ===== METRICS =====
total_mse = 0
total_ssim = 0
count = 0


def denormalize(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


with torch.no_grad():
    for images, _ in dataloader:
        images = images.to(device)

        recon, _, _ = model(images)

        # ===== MSE =====
        mse = F.mse_loss(recon, images, reduction='sum')
        total_mse += mse.item()

        # ===== SSIM =====
        recon_img = denormalize(recon).cpu().numpy()
        real_img = denormalize(images).cpu().numpy()

        for i in range(recon_img.shape[0]):
            # convert CHW → HWC
            r = np.transpose(real_img[i], (1, 2, 0))
            f = np.transpose(recon_img[i], (1, 2, 0))

            s = ssim(r, f, channel_axis=2, data_range=1.0)
            total_ssim += s
            count += 1


# ===== FINAL RESULTS =====
avg_mse = total_mse / len(dataset)
avg_ssim = total_ssim / count

print("\n===== VAE Evaluation Results =====")
print(f"MSE  : {avg_mse:.6f}")
print(f"SSIM : {avg_ssim:.4f}")