"""
Data pipeline: augmentation transforms + stratified train/val/test split.

Expects data laid out in torchvision ImageFolder format:
    sample_data/
        class_a/
            img001.jpg
            img002.jpg
        class_b/
            img001.jpg
            ...
"""

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, IMAGE_SIZE, RANDOM_CROP_PADDING, COLOR_JITTER,
    NORM_MEAN, NORM_STD, BATCH_SIZE, NUM_WORKERS,
    TRAIN_FRAC, VAL_FRAC, TEST_FRAC, RANDOM_SEED,
)


def build_transforms():
    """Returns (train_transform, eval_transform).

    Train transform applies the augmentations called out in the project spec:
    random cropping, horizontal flipping, color jitter, then normalization.
    Eval transform (val/test) only resizes + normalizes -- no stochastic
    augmentation, so validation/test metrics are stable and comparable.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + RANDOM_CROP_PADDING * 2,
                            IMAGE_SIZE + RANDOM_CROP_PADDING * 2)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(**COLOR_JITTER),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    return train_transform, eval_transform


def stratified_split(targets, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC,
                      test_frac=TEST_FRAC, seed=RANDOM_SEED):
    """Stratified indices split so every class keeps the same proportion
    across train/val/test -- important when classes are imbalanced."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    idx = np.arange(len(targets))
    train_idx, temp_idx = train_test_split(
        idx, train_size=train_frac, stratify=targets, random_state=seed
    )
    temp_targets = np.array(targets)[temp_idx]
    relative_val = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=relative_val, stratify=temp_targets, random_state=seed
    )
    return train_idx, val_idx, test_idx


def get_dataloaders():
    """Builds train/val/test DataLoaders from DATA_DIR using ImageFolder,
    with a stratified split and different transforms for train vs eval."""
    train_transform, eval_transform = build_transforms()

    # Load once to get targets for stratification, then wrap with two
    # separately-transformed ImageFolder instances so train gets augmentation
    # and val/test do not, even though they share the same underlying files.
    base = datasets.ImageFolder(DATA_DIR)
    targets = [s[1] for s in base.samples]
    class_names = base.classes

    train_ds_full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    eval_ds_full = datasets.ImageFolder(DATA_DIR, transform=eval_transform)

    train_idx, val_idx, test_idx = stratified_split(targets)

    train_ds = Subset(train_ds_full, train_idx)
    val_ds = Subset(eval_ds_full, val_idx)
    test_ds = Subset(eval_ds_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    return train_loader, val_loader, test_loader, class_names
