import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(data_root="./processed/train", batch_size=64, image_size=32):

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    dataset = datasets.ImageFolder(
        root=data_root,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    print("Total images:", len(dataset))

    return dataloader