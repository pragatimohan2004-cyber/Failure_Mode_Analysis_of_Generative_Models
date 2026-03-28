import torch
import torchvision.utils as vutils
import os
import sys

from pytorch_fid import fid_score

# ================= PATH FIX =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from dcgan_generator import Generator

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================= PATHS =================
real_dir = os.path.join(BASE_DIR, "fid_real")
fake_dir = os.path.join(BASE_DIR, "fid_fake_dcgan")

os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

# ================= CONFIG =================
num_images = 10000
batch_size = 64
nz = 100


# ================= LOAD MODEL =================
netG = Generator(nz=nz).to(device)

model_path = os.path.join(BASE_DIR, "checkpoints", "dcgan_tinyimagenet_generator_final.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Generator not found at: " + model_path)

netG.load_state_dict(torch.load(model_path, map_location=device))
netG.eval()

print("✅ DCGAN Generator loaded")


# ================= SAVE FAKE IMAGES =================
print("\n🎨 Generating fake images...")

count = 0

with torch.no_grad():
    while count < num_images:

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)

        for img in fake:
            vutils.save_image(
                (img + 1) / 2,
                f"{fake_dir}/{count}.png"
            )
            count += 1

            if count >= num_images:
                break

print(f"✅ {count} fake images saved")


# ================= FID COMPUTATION =================
print("\n📊 Computing FID...")

fid_value = fid_score.calculate_fid_given_paths(
    [real_dir, fake_dir],
    batch_size=50,
    device=device,
    dims=2048
)

print(f"\n🔥 DCGAN TinyImageNet FID: {fid_value:.4f}")