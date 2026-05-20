"""
Dataset and data loading utilities for Dog Breed Identification.

This module provides:
- DogBreedDataset: a PyTorch Dataset that loads images from disk and applies transforms
- get_transforms: returns train/val transform pipelines
- create_dataloaders: builds train/val DataLoaders from the labels CSV

The pipeline:  disk (jpg) → Dataset → Transforms → DataLoader → Model
"""

import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class DogBreedDataset(Dataset):
    """
    PyTorch Dataset for dog breed images.

    Each sample is a (image_tensor, label_index) pair.

    Args:
        image_paths: list of Path objects pointing to .jpg files
        labels: list of integer labels (breed index)
        transform: torchvision transform pipeline to apply to each image
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=224):
    """
    Returns separate transform pipelines for training and validation.

    Training transforms include data augmentation (random crops, flips, color
    jitter) to artificially increase dataset variety and reduce overfitting.

    Validation transforms only resize and normalize — no randomness, so
    evaluation is deterministic and comparable across runs.

    Normalization values are ImageNet statistics, which is standard practice
    when using pretrained models.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # e.g. 256 for 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, val_transform


def create_dataloaders(data_dir, batch_size=32, image_size=224, val_split=0.2,
                       num_workers=4, seed=42):
    """
    Builds train and validation DataLoaders from the raw dataset.

    Steps:
    1. Read labels.csv to get (image_id, breed) pairs
    2. Encode breed names as integers (0–119)
    3. Stratified split into train/val (preserving breed proportions)
    4. Wrap in DogBreedDataset with appropriate transforms
    5. Return DataLoaders ready for the training loop

    Args:
        data_dir: path to data/ folder (contains train/, labels.csv)
        batch_size: images per batch (default 32)
        image_size: target image size in pixels (default 224)
        val_split: fraction of data for validation (default 0.2)
        num_workers: parallel data loading workers (default 4)
        seed: random seed for reproducible splits

    Returns:
        train_loader, val_loader, breed_to_idx, idx_to_breed
    """
    data_dir = Path(data_dir)
    labels_df = pd.read_csv(data_dir / "labels.csv")

    # Encode breed names → integer indices
    breeds = sorted(labels_df["breed"].unique())
    breed_to_idx = {breed: i for i, breed in enumerate(breeds)}
    idx_to_breed = {i: breed for breed, i in breed_to_idx.items()}

    # Build lists of (path, label)
    image_paths = [data_dir / "train" / f"{row['id']}.jpg" for _, row in labels_df.iterrows()]
    labels = [breed_to_idx[row["breed"]] for _, row in labels_df.iterrows()]

    # Stratified train/val split — keeps breed proportions equal in both sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )

    # Create transform pipelines
    train_transform, val_transform = get_transforms(image_size)

    # Build datasets
    train_dataset = DogBreedDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = DogBreedDataset(val_paths, val_labels, transform=val_transform)

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    print(f"Breeds: {len(breeds)} | Batch size: {batch_size} | Image size: {image_size}x{image_size}")

    return train_loader, val_loader, breed_to_idx, idx_to_breed
