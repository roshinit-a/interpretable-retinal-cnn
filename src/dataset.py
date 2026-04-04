"""
dataset.py — DataLoader factory for the Kaggle Retinal OCT dataset.

The Kaggle OCT2017 dataset (Zhang et al., 2018) contains 84,484 validated OCT
images across four classes:
    CNV   — Choroidal Neovascularisation  (fluid beneath retina)
    DME   — Diabetic Macular Edema        (intraretinal fluid)
    DRUSEN — Age-related drusen deposits
    NORMAL — Healthy retina

Reference:
    Kermany, D. S. et al. (2018). Identifying Medical Diagnoses and Treatable
    Diseases by Image-Based Deep Learning. Cell, 172(5), 1122-1131.e9.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections import Counter
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ──────────────────────────────────────────────────────────────────────────────
#  ImageNet statistics used for normalisation.
#  Using ImageNet mean/std is standard practice even for medical images trained
#  from scratch — it centres pixel distributions and stabilises early training.
# ──────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE    = 224   # ViT/ResNet standard; also minimal for meaningful Grad-CAM
BATCH_SIZE    = 32    # Fits comfortably on a single 8 GB GPU


def get_transforms(split: str) -> transforms.Compose:
    """
    Build the augmentation / pre-processing pipeline for a given split.

    Args:
        split: One of 'train', 'val', or 'test'.

    Returns:
        A torchvision Compose pipeline appropriate for the specified split.

    Design rationale for training augmentations:
        - RandomHorizontalFlip: OCT B-scans are bilaterally symmetric; flipping
          does not change pathological semantics.
        - RandomRotation(10°): mimics minor head-tilt variations during imaging;
          kept small to avoid distorting layer topology.
        - ColorJitter(brightness=0.2): simulates variable illumination and sensor
          gain differences across OCT machines.
        - NO vertical flip: retinal layers have strict top-bottom anatomy (ILM at
          top, RPE at bottom); flipping vertically produces anatomically invalid
          images that confuse the model.
        - NO large crops: OCT pathology often resides near the fovea; aggressive
          cropping risks excising the diagnostically relevant region.
    """
    if split == "train":
        return transforms.Compose([
            # Resize first to avoid augmenting at full resolution (saves compute)
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # ── Augmentations ────────────────────────────────────────────────
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.1),
            # ── Tensor conversion + normalisation ────────────────────────────
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # Val / test: deterministic pipeline — only resize + normalise.
        # No augmentation at test time; we want to measure true generalisation.
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def _print_class_distribution(dataset: datasets.ImageFolder, split_name: str) -> None:
    """
    Print the number of samples per class for a given split.

    Args:
        dataset:    ImageFolder dataset whose .targets we inspect.
        split_name: Human-readable label, e.g. 'Train'.
    """
    counts = Counter(dataset.targets)
    total  = sum(counts.values())
    print(f"\n{split_name} split — {total} samples total:")
    for idx, class_name in enumerate(dataset.classes):
        n = counts[idx]
        bar = "█" * (n // 500)       # visual bar at 1 block per 500 samples
        print(f"  {class_name:<10} {n:>6}  ({100 * n / total:5.1f}%)  {bar}")


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build and return train, val, and test DataLoaders for the OCT dataset.

    Expected directory layout (mirrors the Kaggle download):
        data_dir/
            train/
                CNV/     *.jpeg
                DME/     *.jpeg
                DRUSEN/  *.jpeg
                NORMAL/  *.jpeg
            test/
                CNV/     ...
                DME/     ...
                DRUSEN/  ...
                NORMAL/  ...

    Args:
        data_dir:    Root directory containing 'train/' and 'test/' sub-folders.
        batch_size:  Samples per minibatch.  Default 32.
        val_split:   Fraction of the training set to hold out for validation.
                     Default 0.1 (≈ 10% of train).
        num_workers: DataLoader worker processes.  Set to 0 on Windows.
        seed:        RNG seed for the train/val random split.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        FileNotFoundError: If data_dir does not contain 'train/' or 'test/'.
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    for d in (train_dir, test_dir):
        if not d.is_dir():
            raise FileNotFoundError(
                f"Expected directory not found: {d}\n"
                "Please follow data/README.md to download and arrange the dataset."
            )

    # ── Load full training set with augmentation pipeline ────────────────────
    full_train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=get_transforms("train"),
    )

    # ── Load a second view of the training data for validation ────────────────
    # We use the same folder but a different (val) transform.  The random_split
    # indices are shared, so there is no data leakage.
    full_val_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=get_transforms("val"),
    )

    # ── Deterministic train / val split ──────────────────────────────────────
    n_total = len(full_train_dataset)
    n_val   = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    # Wrap with Subset — each subset uses its respective transform
    from torch.utils.data import Subset
    train_dataset = Subset(full_train_dataset, train_indices.indices)
    val_dataset   = Subset(full_val_dataset,   val_indices.indices)

    # ── Test set  ─────────────────────────────────────────────────────────────
    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=get_transforms("test"),
    )

    # ── Print class distributions for sanity checking ────────────────────────
    _print_class_distribution(full_train_dataset, "Train (full)")
    _print_class_distribution(test_dataset,       "Test")
    print(f"\n  Train subset : {len(train_dataset)} samples")
    print(f"  Val   subset : {len(val_dataset)} samples")

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    # pin_memory=True moves tensors to pinned (page-locked) host memory so GPU
    # transfers are faster during training.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # re-shuffle each epoch for stochastic gradient
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,          # deterministic evaluation
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Expose class names on the loaders for convenience in other modules
    train_loader.dataset.classes = full_train_dataset.classes   # type: ignore[attr-defined]
    test_loader.dataset.classes  = test_dataset.classes          # type: ignore[attr-defined]

    print(f"\nClass order: {full_train_dataset.classes}")
    print("DataLoaders ready.\n")

    return train_loader, val_loader, test_loader


# ── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "data"
    train_l, val_l, test_l = get_dataloaders(data_root)

    # Grab one batch and confirm shapes
    images, labels = next(iter(train_l))
    print(f"Batch shape : {images.shape}")   # expect [32, 3, 224, 224]
    print(f"Labels      : {labels[:8]}")
