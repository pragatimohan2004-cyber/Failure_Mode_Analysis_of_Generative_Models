import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

data_dir = "./tiny-imagenet-200"

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir,"train"),
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4
)

print("Total training images:", len(train_dataset))

for images, labels in train_loader:
    print("Batch shape:", images.shape)
    print("Labels shape:", labels.shape)
    break