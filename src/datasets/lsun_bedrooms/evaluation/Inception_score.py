import os
import sys

# ================= PATH FIX =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, ".."))

import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from PIL import Image
from scipy.stats import entropy

from model.sngan_generator import Generator


# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ================= CONFIG =================
nz = 100
num_images = 10000
gen_batch_size = 64
eval_batch_size = 32   # 🔥 IMPORTANT (prevents OOM)


# ================= PATHS =================
fake_dir = os.path.join(BASE_DIR, "is_fake_sngan")

checkpoint_path = os.path.join(
    BASE_DIR,
    "..",
    "training",
    "checkpoints_lsun",
    "generator_final.pth"
)

os.makedirs(fake_dir, exist_ok=True)


# ================= CHECK CHECKPOINT =================
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"❌ Generator not found at:\n{checkpoint_path}")


# ================= LOAD GENERATOR =================
print("Loading generator...")
netG = Generator(nz=nz).to(device)
netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
netG.eval()


# ================= GENERATE IMAGES =================
print("Generating images for IS...")

count = 0

with torch.no_grad():
    while count < num_images:
        noise = torch.randn(gen_batch_size, nz, 1, 1, device=device)
        fake = netG(noise)

        for img in fake:
            img_path = os.path.join(fake_dir, f"{count}.png")
            transforms.ToPILImage()((img + 1) / 2).save(img_path)

            count += 1
            if count >= num_images:
                break

print(f"✅ Generated {num_images} images")


# ================= LOAD FILE LIST =================
files = os.listdir(fake_dir)

if len(files) == 0:
    raise RuntimeError("❌ No images found in fake_dir!")

print(f"Total images found: {len(files)}")


# ================= LOAD INCEPTION =================
print("Loading InceptionV3...")

model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
model.eval()


# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])


# ================= BATCHED INFERENCE =================
print("Running Inception predictions (memory-safe)...")

preds_list = []

with torch.no_grad():
    for i in range(0, len(files), eval_batch_size):

        batch_imgs = []

        for j in range(i, min(i + eval_batch_size, len(files))):
            try:
                img = Image.open(os.path.join(fake_dir, files[j])).convert("RGB")
                img = transform(img)
                batch_imgs.append(img)
            except:
                continue

        if len(batch_imgs) == 0:
            continue

        batch = torch.stack(batch_imgs).to(device)

        preds = model(batch)
        preds = torch.nn.functional.softmax(preds, dim=1)

        preds_list.append(preds.cpu())

# 🔥 combine all batches
preds = torch.cat(preds_list, dim=0).numpy()


# ================= IS CALC =================
print("Calculating Inception Score...")

split_size = 10
scores = []

N = preds.shape[0]

for i in range(split_size):
    part = preds[i * N // split_size:(i + 1) * N // split_size]
    py = np.mean(part, axis=0)

    scores.append(np.exp(np.mean([entropy(p, py) for p in part])))

is_mean = np.mean(scores)
is_std = np.std(scores)


# ================= OUTPUT =================
print(f"\n SNGAN LSUN Inception Score: {is_mean:.4f} ± {is_std:.4f}")