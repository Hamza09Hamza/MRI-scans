import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_transforms(config):
    """Create train and test transforms with augmentation."""
    img_size = config["data"]["img_size"]
    aug = config["augmentation"]

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5 if aug["horizontal_flip"] else 0.0),
        transforms.RandomRotation(aug["rotation_degrees"]),
        transforms.ColorJitter(
            brightness=aug["brightness_jitter"],
            contrast=aug["contrast_jitter"]
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, test_transforms

def get_dataloaders(config, batch_size=None):
    """Create optimized train and test dataloaders for M1 Pro."""
    data_dir = os.path.dirname(os.path.abspath(__file__)).replace("utils", "")
    train_dir = os.path.join(data_dir, config["data"]["train_dir"])
    test_dir = os.path.join(data_dir, config["data"]["test_dir"])

    train_transforms, test_transforms = get_transforms(config)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    bs = batch_size or config["training"]["batch_size"]
    device = config["device"]

    # OPTIMIZATION: pin_memory is False for MPS; num_workers increased for CPU saturation[cite: 14, 15]
    num_workers = 0
    pin_memory = True if device == "cuda" else False # Only True for NVIDIA[cite: 14]

    train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    test_loader = DataLoader(
            test_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, test_loader, train_dataset.classes