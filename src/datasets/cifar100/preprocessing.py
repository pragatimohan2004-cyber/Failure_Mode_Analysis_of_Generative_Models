import pickle
import numpy as np
from PIL import Image
import os

dataset_path = "../CIFAR_100/cifar-100-python"
output_path = "../CIFAR_100/processed"


def process_split(split_name):
    print(f"\n🚀 Processing {split_name} images...")

    with open(os.path.join(dataset_path, split_name), "rb") as f:
        data = pickle.load(f, encoding="bytes")

    images = data[b'data']
    labels = data[b'fine_labels']

    total = len(images)

    for i, (img, label) in enumerate(zip(images, labels)):

        # Create class folder
        class_dir = os.path.join(output_path, split_name, str(label))
        os.makedirs(class_dir, exist_ok=True)

        # Reshape and convert
        img = img.reshape(3, 32, 32).transpose(1, 2, 0)

        # Ensure uint8 format (important!)
        img = img.astype(np.uint8)

        # Save image
        save_path = os.path.join(class_dir, f"{i}.png")
        Image.fromarray(img).save(save_path)

        # Progress log (every 5000 images)
        if i % 5000 == 0:
            print(f"{split_name}: {i}/{total} images saved")

    print(f"✅ {split_name.capitalize()} images completed ({total} images)")


# ================= RUN =================
os.makedirs(output_path, exist_ok=True)

process_split("train")
process_split("test")

print("\CIFAR-100 preprocessing complete!")