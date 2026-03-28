import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, image_size=64):
        self.image_paths = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.image_paths.append(os.path.join(root, file))

        print(f"Total images found: {len(self.image_paths)}")

        if len(self.image_paths) == 0:
            raise RuntimeError("No images found!")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def get_dataloader(root_dir, batch_size=64):
    dataset = TinyImageNetDataset(root_dir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )