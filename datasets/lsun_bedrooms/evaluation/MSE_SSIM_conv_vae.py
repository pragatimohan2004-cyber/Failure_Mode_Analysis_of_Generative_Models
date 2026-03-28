import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

from model.convolutional_vae_model import ConvVAE_LSUN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= DATASET =================
class LSUNDataset(Dataset):
    def __init__(self, folder, max_images=5000):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith((".jpg", ".png"))
        ][:max_images]

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ================= LOAD DATA =================
dataset = LSUNDataset("../data/lsun_clean", max_images=5000)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)


# ================= LOAD MODEL =================
model = ConvVAE_LSUN().to(device)
model.load_state_dict(torch.load("../training/checkpoints_vae/vae_lsun_final.pth"))
model.eval()

print("✅ Model loaded")


# ================= EVALUATION =================
total_mse = 0
total_ssim = 0
count = 0

with torch.no_grad():
    for images in dataloader:

        images = images.to(device)

        recon, _, _ = model(images)

        # ===== MSE =====
        mse = F.mse_loss(recon, images, reduction='mean')
        total_mse += mse.item()

        # ===== SSIM =====
        recon_np = recon.cpu().numpy()
        images_np = images.cpu().numpy()

        for i in range(images.size(0)):
            ssim_val = ssim(
                images_np[i].transpose(1,2,0),
                recon_np[i].transpose(1,2,0),
                channel_axis=2,
                data_range=2  # because normalized [-1,1]
            )
            total_ssim += ssim_val
            count += 1


# ================= RESULTS =================
avg_mse = total_mse / len(dataloader)
avg_ssim = total_ssim / count

print("\n📊 LSUN VAE Evaluation Results")
print(f"MSE  : {avg_mse:.6f}")
print(f"SSIM : {avg_ssim:.4f}")