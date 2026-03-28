import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

from dataloader_tinyimagenet import get_dataloader
from convolutional_vae_model import ConvVAE_Tiny


# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== DATA =====
dataloader = get_dataloader("tiny-imagenet-200/val", batch_size=64)


# ===== MODEL =====
model = ConvVAE_Tiny().to(device)   # ✅ FIXED (Device → device)

model.load_state_dict(
    torch.load("./checkpoints_vae_tiny/vae_final.pth", map_location=device)
)

model.eval()


# ===== METRICS =====
total_mse = 0
total_ssim = 0
count = 0
num_batches = 0


with torch.no_grad():
    for batch in dataloader:

        # ✅ HANDLE DATALOADER OUTPUT
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = images.to(device)

        # ===== FORWARD =====
        recon, _, _ = model(images)

        # ===== MSE =====
        mse = F.mse_loss(recon, images, reduction='mean')
        total_mse += mse.item()
        num_batches += 1

        # ===== SSIM =====
        recon_np = recon.cpu().numpy()
        images_np = images.cpu().numpy()

        for i in range(images.size(0)):
            total_ssim += ssim(
                images_np[i].transpose(1, 2, 0),
                recon_np[i].transpose(1, 2, 0),
                channel_axis=2,
                data_range=2  # because [-1,1]
            )
            count += 1


# ===== FINAL RESULTS =====
avg_mse = total_mse / num_batches
avg_ssim = total_ssim / count

print("\n📊 TinyImageNet VAE Results")
print(f"MSE  : {avg_mse:.6f}")
print(f"SSIM : {avg_ssim:.4f}")