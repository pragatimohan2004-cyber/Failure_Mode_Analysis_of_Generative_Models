import os
from PIL import Image
from tqdm import tqdm

# absolute-safe paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

input_root = os.path.join(BASE_DIR, "lsun_300k")
output_root = os.path.join(BASE_DIR, "lsun_clean")

image_size = 64
total = 0
corrupted = 0

os.makedirs(output_root, exist_ok=True)

print("\nStarting LSUN preprocessing...\n")
print("Reading images from:", input_root)

images = [
    f for f in os.listdir(input_root)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print("Images found:", len(images))

for img_name in tqdm(images):

    img_path = os.path.join(input_root, img_name)

    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)

        save_path = os.path.join(output_root, img_name)
        img.save(save_path)

        total += 1

    except Exception:
        corrupted += 1


print("\nPreprocessing finished.")
print("Total processed images:", total)
print("Corrupted images removed:", corrupted)