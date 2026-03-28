import os
import sys

# ================= PATH FIX =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))  # 🔥 fix import

import torch
import torchvision.utils as vutils
from pytorch_fid import fid_score

from model.sngan_generator import Generator

# ================= CONFIG =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nz = 100
num_images = 10000

# ================= PATHS =================
real_dir = os.path.join(BASE_DIR, "fid_real")
fake_dir = os.path.join(BASE_DIR, "fid_fake_sngan")

# 🔥 FIXED CHECKPOINT PATH
checkpoint_path = os.path.join(BASE_DIR, "..", "training", "checkpoints_lsun", "generator_final.pth")
os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

# ================= LOAD GENERATOR =================
print("Loading generator...")
netG = Generator(nz=nz).to(device)

netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
netG.eval()

# ================= GENERATE FAKE IMAGES =================
print("Generating fake images...")
count = 0

with torch.no_grad():
    while count < num_images:
        noise = torch.randn(64, nz, 1, 1, device=device)
        fake = netG(noise)

        for img in fake:
            vutils.save_image(
                (img + 1) / 2,  # normalize [-1,1] → [0,1]
                f"{fake_dir}/{count}.png"
            )
            count += 1
            if count >= num_images:
                break

print("Fake images generated!")

# ================= CHECK REAL IMAGES =================
if len(os.listdir(real_dir)) == 0:
    raise RuntimeError("❌ fid_real folder is EMPTY — add real images!")

# ================= COMPUTE FID =================
print("Calculating FID...")

fid_value = fid_score.calculate_fid_given_paths(
    [real_dir, fake_dir],
    batch_size=64,
    device=device,
    dims=2048
)

print(f"\n SNGAN LSUN FID: {fid_value:.4f}")