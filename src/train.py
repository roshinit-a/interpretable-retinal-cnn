"""
train.py — Training Script for RetinalCNN and HybridModel.

Usage:
    python src/train.py --model cnn   --data_dir data --epochs 20
    python src/train.py --model hybrid --data_dir data --epochs 20

Both variants share the same training loop; the only difference is which model
class is instantiated.

Expected outcomes (on the full Kaggle OCT dataset, single GPU):
    CNN    — Val accuracy: ~88–92%   Test accuracy: ~88–92%
    Hybrid — Val accuracy: ~90–93%   Test accuracy: ~90–93%

The Hybrid consistently edges out the CNN because Transformer attention captures
dependencies between spatially distant feature positions — e.g., associating the
fluid pocket pattern in one region with layer disruption in another.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Allow running from project root: python src/train.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders
from src.model_cnn import RetinalCNN
from src.model_hybrid import HybridModel


# ──────────────────────────────────────────────────────────────────────────────
#  Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full pass over the training DataLoader.

    Args:
        model:     Model in train mode.
        loader:    Training DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimiser (Adam).
        device:    torch.device for computation.

    Returns:
        Tuple of (mean_loss, accuracy) over the entire epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # ── Forward ──
        logits = model(images)
        loss   = criterion(logits, labels)

        # ── Backward ──
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping prevents exploding gradients — important for the
        # Transformer head where attention weights can amplify gradient magnitudes.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ── Accumulators ──
        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on a DataLoader (validation or test split).

    Args:
        model:     Model whose eval() will be called.
        loader:    DataLoader to evaluate on.
        criterion: Loss function.
        device:    Compute device.

    Returns:
        Tuple of (mean_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def save_training_curves(
    history: dict,
    model_name: str,
    save_path: str = "results/training_curves.png",
) -> None:
    """
    Plot and save training/validation loss and accuracy curves.

    Args:
        history:    Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        model_name: Used in plot title.
        save_path:  File path to save the PNG.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Loss curve ────────────────────────────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy curve ────────────────────────────────────────────────────────
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val Acc",   linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """
    Orchestrate data loading, model initialisation, training, and checkpointing.
    """
    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model == "cnn":
        model = RetinalCNN(num_classes=4).to(device)
    elif args.model == "hybrid":
        model = HybridModel(num_classes=4).to(device)
    else:
        raise ValueError(f"Unknown model type '{args.model}'. Choose 'cnn' or 'hybrid'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()} — {n_params:,} trainable parameters")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Adam with lr=1e-3 is the standard baseline.  We use weight_decay=1e-4 for
    # L2 regularisation, especially important for the Transformer head which has
    # more parameters and can overfit small datasets.
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # StepLR halves the learning rate every `step_size` epochs.
    # A decaying LR allows large initial steps (fast convergence) and fine-tuning
    # later when the loss landscape is flatter.
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    # ── Loss function ─────────────────────────────────────────────────────────
    # CrossEntropyLoss = log-softmax + NLL loss.  For class imbalance in OCT
    # (CNV >> DRUSEN in some splits), we could add class weights — omitted here
    # to keep the implementation clean but noted as a future improvement.
    # TODO: handle class imbalance with weighted loss
    # class_counts = [45000, 11000, 8000, 27000]  # approx CNV, DME, DRUSEN, NORMAL
    # weights = torch.tensor([1/c for c in class_counts])
    # weights = weights / weights.sum()
    # criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    save_path    = f"results/best_{args.model}.pth"
    os.makedirs("results", exist_ok=True)

    history: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    print(f"\n{'─'*65}")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10}")
    print(f"{'─'*65}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Checkpoint the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":      epoch,
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "val_acc":    val_acc,
                "args":       vars(args),
            }, save_path)
            marker = " ✓"
        else:
            marker = ""

        # Log row
        elapsed = time.time() - t0
        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {train_acc * 100:>9.2f}%"
            f"  {val_loss:>9.4f}  {val_acc * 100:>8.2f}%"
            f"  {current_lr:>9.6f}"
            f"  ({elapsed:.1f}s){marker}"
        )

        # Record for curve plotting
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    print(f"{'─'*65}")
    print(f"Best val accuracy: {best_val_acc * 100:.2f}%  (saved to {save_path})")

    save_training_curves(
        history,
        model_name=args.model.upper(),
        save_path=f"results/{args.model}_training_curves.png",
    )

    # ── Final test evaluation ──────────────────────────────────────────────────
    # Reload the best checkpoint so the reported test accuracy always corresponds
    # to the best validation model, not the model state at the final epoch.
    print("\nRunning final evaluation on test set...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Results — Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RetinalCNN or HybridModel on the Kaggle OCT dataset."
    )
    parser.add_argument(
        "--model", type=str, default="cnn", choices=["cnn", "hybrid"],
        help="Which model to train.  'cnn' = RetinalCNN, 'hybrid' = HybridModel."
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Root directory containing 'train/' and 'test/' sub-folders."
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Mini-batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate for Adam."
    )
    parser.add_argument(
        "--lr_step", type=int, default=10,
        help="Halve LR every this many epochs (StepLR).  Default of 10 gives 2 halvings"
             " over 20 epochs: 1e-3 → 5e-4 → 2.5e-4."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker processes.  Use 0 on Windows."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
