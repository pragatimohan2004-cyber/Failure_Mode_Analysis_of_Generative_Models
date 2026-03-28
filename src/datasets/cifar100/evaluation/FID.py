import torch
from pytorch_fid import fid_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 IMPORTANT PATHS
real_path = "metrics_cifar100/fid_real"     # same as DCGAN
fake_path = "generated_sngan"               # SNGAN output

print("Computing CIFAR-100 SNGAN FID...")

fid = fid_score.calculate_fid_given_paths(
    [real_path, fake_path],
    batch_size=50,
    device=device,
    dims=2048
)

print(f"\n🔥 CIFAR-100 SNGAN FID: {fid:.2f}")

with open("metrics_cifar100/fid_results.txt", "a") as f:
    f.write(f"CIFAR100 SNGAN FID: {fid:.2f}\n")