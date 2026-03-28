import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class LSUNBedroomDataset(Dataset):

    def __init__(self, root_dir, image_size=64):

        self.root_dir = root_dir
        self.image_paths = []

        # ✅ FIX 1: recursive loading (important for LSUN)
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(root, file))

        print("Total LSUN images found:", len(self.image_paths))

        # ✅ Safety check
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

        # ✅ Transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"Error loading image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        return img


# ✅ FIX 2: REQUIRED for training script
def get_dataloader(root_dir, batch_size=64, image_size=64, num_workers=2):

    dataset = LSUNBedroomDataset(root_dir, image_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader


# ✅ Optional standalone test
if __name__ == "__main__":

    root = "./lsun_clean"   # change if needed

    loader = get_dataloader(root, batch_size=8)

    print("Total images:", len(loader.dataset))

    sample = next(iter(loader))

    print("Batch shape:", sample.shape)