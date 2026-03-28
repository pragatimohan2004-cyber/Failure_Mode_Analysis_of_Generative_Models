import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageFolderDataset(Dataset):
    def __init__(self, folder):
        self.paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

fake_path = "../generated_sngan"

dataset = ImageFolderDataset(fake_path)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
inception.eval()

print("Calculating Inception Score...")

preds = []

with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        output = inception(batch)
        preds.append(F.softmax(output, dim=1).cpu().numpy())

preds = np.concatenate(preds, axis=0)

splits = 10
scores = []

for i in range(splits):
    part = preds[i * len(preds)//splits : (i+1) * len(preds)//splits]

    py = np.mean(part, axis=0)

    kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
    kl = np.sum(kl, axis=1)

    scores.append(np.exp(np.mean(kl)))

mean, std = np.mean(scores), np.std(scores)

print(f"\n CIFAR-100 SNGAN IS: {mean:.2f} ± {std:.2f}")

with open("inception_results.txt", "a") as f:
    f.write(f"CIFAR100 SNGAN IS: {mean:.2f} ± {std:.2f}\n")