import torch
import torchvision.utils as vutils
import os
import sys
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from ladder_vae_model import LadderVAE
from dataloader_tinyimagenet import get_dataloader


# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== LOAD MODEL =====
model = LadderVAE().to(device)
model.load_state_dict(torch.load("../checkpoints_lvae/ladder_vae_tiny.pth", map_location=device))
model.eval()


# ===== DATA =====
dataloader = get_dataloader(
    root_dir="tiny-imagenet-200/val",
    batch_size=32
)


# ===== OUTPUT DIR =====
os.makedirs("../vae_samples", exist_ok=True)


# ===== METRICS =====
total_mse = 0
total_ssim = 0
count = 0


with torch.no_grad():
    for i, images in enumerate(dataloader):

        images = images.to(device)

        # ✅ FIXED unpacking
        recon, _= model(images)

        # ===== MSE =====
        mse = F.mse_loss(recon, images, reduction='mean')
        total_mse += mse.item()

        # ===== SSIM =====
        recon_np = recon.cpu().numpy()
        images_np = images.cpu().numpy()

        for j in range(images.size(0)):
            total_ssim += ssim(
                images_np[j].transpose(1, 2, 0),
                recon_np[j].transpose(1, 2, 0),
                channel_axis=2,
                data_range=2  # because normalized [-1,1]
            )
            count += 1

        # ===== SAVE IMAGES =====
        vutils.save_image(
            (recon + 1) / 2,  # denormalize
            f"../vae_samples/recon_{i}.png"
        )

        if i == 5:
            break


# ===== FINAL RESULTS =====
avg_mse = total_mse / (i + 1)
avg_ssim = total_ssim / count

print("\n📊 Ladder VAE Evaluation Results")
print(f"MSE  : {avg_mse:.6f}")
print(f"SSIM : {avg_ssim:.4f}")