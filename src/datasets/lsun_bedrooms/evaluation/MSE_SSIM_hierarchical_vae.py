import torch
import torch.nn as nn
import os
import sys
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from skimage.metrics import structural_similarity as ssim
import numpy as np

# ================= PATH FIX =================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from model.ladder_vae import LadderVAE
from data.dataloader_lsun import LSUNBedroomDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= CONFIG =================
batch_size = 16   # lower for eval
model_path = "../checkpoints_lvae/ladder_vae_lsun.pth"


# ================= DATA =================
dataset = LSUNBedroomDataset("../data/lsun_clean")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ================= MODEL =================
model = LadderVAE().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("✅ Model loaded!")


# ================= METRICS =================
mse_total = 0
ssim_total = 0
count = 0

# folder to save images
os.makedirs("../generated_lvae", exist_ok=True)


print("\n🚀 Evaluating Ladder VAE...\n")


with torch.no_grad():

    for i, images in enumerate(dataloader):

        images = images.to(device)

        recon, _, _ = model(images)

        # ================= MSE =================
        mse = nn.functional.mse_loss(recon, images, reduction="mean")
        mse_total += mse.item()

        # ================= SSIM =================
        images_np = images.cpu().numpy()
        recon_np = recon.cpu().numpy()

        for j in range(images_np.shape[0]):
            real_img = np.transpose(images_np[j], (1, 2, 0))
            fake_img = np.transpose(recon_np[j], (1, 2, 0))

            ssim_val = ssim(
                real_img,
                fake_img,
                channel_axis=2,
                data_range=1.0
            )

            ssim_total += ssim_val
            count += 1

        # ================= SAVE SAMPLE =================
        if i == 0:
            comparison = torch.cat([images[:8], recon[:8]])
            vutils.save_image(
                comparison,
                "../generated_lvae/reconstruction.png",
                nrow=8,
                normalize=True
            )


# ================= FINAL RESULTS =================
avg_mse = mse_total / len(dataloader)
avg_ssim = ssim_total / count

print("\n====== Ladder VAE Evaluation ======")
print(f"MSE  : {avg_mse:.4f}")
print(f"SSIM : {avg_ssim:.4f}")


# ================= SAVE RESULTS =================
with open("../generated_lvae/results.txt", "w") as f:
    f.write(f"MSE: {avg_mse:.4f}\n")
    f.write(f"SSIM: {avg_ssim:.4f}\n")

print("\n✅ Results saved!")