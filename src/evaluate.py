"""
evaluate.py — Model Evaluation, Confusion Matrix, and Interpretability (SHAP).

This script loads a saved checkpoint, runs it on the held-out test set, and
produces three complementary views of model behaviour:

  1. Per-class precision / recall / F1 (sklearn)
     → Tells us *which* classes are confused, not just overall accuracy.

  2. Confusion matrix heatmap (matplotlib + seaborn)
     → Visual distribution of prediction errors across all class pairs.

  3. SHAP DeepExplainer values (shap library)
     → Pixel-level feature attribution from an expected-value baseline.

──────────────────────────────────────────────────────────────────────────────
  Grad-CAM vs SHAP — When to Use Which
──────────────────────────────────────────────────────────────────────────────

  ┌──────────────┬────────────────────────────┬────────────────────────────┐
  │ Dimension    │ Grad-CAM                   │ SHAP (DeepExplainer)       │
  ├──────────────┼────────────────────────────┼────────────────────────────┤
  │ What it      │ Gradient × activation in   │ Shapley values: how much   │
  │ measures     │ the last conv layer.        │ each input pixel shifts    │
  │              │ "Which spatial regions      │ the model's output away    │
  │              │  drive this prediction?"    │ from a reference baseline. │
  ├──────────────┼────────────────────────────┼────────────────────────────┤
  │ Output       │ Single low-res heatmap per  │ Signed attribution map     │
  │ format       │ image (upsampled to orig.)  │ (same size as input, + / -)│
  ├──────────────┼────────────────────────────┼────────────────────────────┤
  │ Mathematical │ Gradient-weighted activation│ Exact Shapley game-theory  │
  │ grounding    │ (approximation, not exact)  │ axioms (exact for deep nets│
  │              │                             │ via linearisation)         │
  ├──────────────┼────────────────────────────┼────────────────────────────┤
  │ When to use  │ Quick sanity check during   │ When you need rigorous,    │
  │              │ development; when you want  │ game-theoretically fair     │
  │              │ a *class-discriminative*    │ attributions; regulatory   │
  │              │ saliency map in one pass.   │ / clinical reporting.      │
  ├──────────────┼────────────────────────────┼────────────────────────────┤
  │ Computational│ ~1 forward + 1 backward     │ Many forward passes over   │
  │ cost         │ pass per image.  Fast.      │ background samples.  Slow. │
  ├──────────────┼────────────────────────────┼────────────────────────────┤
  │ Limitations  │ Coarse (heatmap resolution  │ Requires a background      │
  │              │ = feature map size);        │ dataset; can be noisy for  │
  │              │ class-score sensitive.       │ deep non-linear networks.  │
  └──────────────┴────────────────────────────┴────────────────────────────┘

  Clinical recommendation: use Grad-CAM for interactive exploration and SHAP
  for audit trails and regulatory submissions.

Reference:
    Lundberg, S. M., & Lee, S.-I. (2017).  A Unified Approach to Interpreting
    Model Predictions.  NeurIPS.  https://arxiv.org/abs/1705.07874
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import get_dataloaders
from src.model_cnn import RetinalCNN
from src.model_hybrid import HybridModel


# ──────────────────────────────────────────────────────────────────────────────
#  Core evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model inference over an entire DataLoader.

    Args:
        model:  Model in eval mode.
        loader: DataLoader (typically the test split).
        device: Compute device.

    Returns:
        Tuple of (all_preds, all_labels) as numpy int arrays, shape [N].
    """
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def print_classification_report(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> None:
    """
    Print per-class precision, recall, F1, and macro/weighted averages.

    Args:
        preds:       Predicted class indices [N].
        labels:      Ground-truth class indices [N].
        class_names: Ordered list of class name strings.
    """
    print("\n" + "─" * 60)
    print("Classification Report")
    print("─" * 60)
    print(classification_report(labels, preds, target_names=class_names, digits=4))


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    save_path: str = "results/confusion_matrix.png",
) -> None:
    """
    Compute and save a normalised confusion matrix heatmap.

    Normalisation is row-wise (recall-normalised): each cell shows the fraction
    of true-class samples predicted as each class.  This makes class-level errors
    immediately visible regardless of class imbalance.

    Args:
        preds:       Predicted class indices [N].
        labels:      Ground-truth class indices [N].
        class_names: Ordered list of class name strings.
        save_path:   File path to save the PNG.
    """
    os.makedirs(os.path.dirname(save_path) or "results", exist_ok=True)

    cm = confusion_matrix(labels, preds)
    # Row-normalise: cm_norm[i, j] = fraction of class i predicted as class j
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class",      fontsize=12)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  SHAP DeepExplainer
# ──────────────────────────────────────────────────────────────────────────────

def run_shap_explanations(
    model: nn.Module,
    test_loader,
    device: torch.device,
    n_background: int = 50,
    n_explain: int = 20,
    save_path: str = "results/shap_summary.png",
) -> None:
    """
    Compute SHAP DeepExplainer values and save a summary plot.

    SHAP DeepExplainer uses a background dataset to define the "expected" model
    output.  For each test image, it computes a signed attribution map showing
    which pixels push the prediction toward or away from each class relative to
    that expected baseline.

    Args:
        model:        Trained model in eval mode.
        test_loader:  Test DataLoader.
        device:       Compute device.  Note: SHAP DeepExplainer runs on CPU for
                      stability; we move tensors accordingly.
        n_background: Number of background samples used to estimate expected output.
                      50 is a good trade-off between variance and compute time.
        n_explain:    Number of test samples to explain.
        save_path:    File path for the SHAP summary PNG.
    """
    os.makedirs(os.path.dirname(save_path) or "results", exist_ok=True)

    # Move model to CPU — SHAP DeepExplainer has known CUDA tensor-sharing issues
    model_cpu = model.to("cpu").eval()

    # Collect background and explanation batches
    background_batches, explain_batches = [], []
    n_bg_collected, n_ex_collected = 0, 0

    for images, _ in test_loader:
        # Background: first n_background images
        if n_bg_collected < n_background:
            need = min(images.size(0), n_background - n_bg_collected)
            background_batches.append(images[:need])
            n_bg_collected += need

        # Explanation: next n_explain images
        if n_ex_collected < n_explain and n_bg_collected >= n_background:
            need = min(images.size(0), n_explain - n_ex_collected)
            explain_batches.append(images[:need])
            n_ex_collected += need

        if n_bg_collected >= n_background and n_ex_collected >= n_explain:
            break

    background = torch.cat(background_batches, dim=0)[:n_background]
    test_imgs  = torch.cat(explain_batches,    dim=0)[:n_explain]

    print(f"Running SHAP DeepExplainer on {n_explain} test samples "
          f"with {n_background} background samples …")

    # ── SHAP DeepExplainer ────────────────────────────────────────────────────
    # DeepExplainer approximates Shapley values using a linearisation of the
    # model around the expected value computed over the background set.
    # This is much faster than KernelSHAP (which treats the model as a black box)
    # while still satisfying the SHAP consistency and efficiency axioms.
    explainer  = shap.DeepExplainer(model_cpu, background)
    shap_values = explainer.shap_values(test_imgs)
    # shap_values: list of [n_explain, 3, 224, 224] — one array per class

    # ── Summary plot ─────────────────────────────────────────────────────────
    # SHAP image_plot expects numpy arrays with shape [N, H, W, C]
    test_imgs_np = test_imgs.permute(0, 2, 3, 1).numpy()    # [N, H, W, 3]

    # For a multi-class model, shap.image_plot shows per-class attribution grids.
    # We plot for all 4 classes stacked vertically.
    shap_np = [sv.transpose(0, 2, 3, 1) for sv in shap_values]   # [N, H, W, 3]

    fig_width  = max(n_explain * 1.5, 12)
    fig_height = 4 * 4   # 4 classes
    fig = plt.figure(figsize=(fig_width, fig_height))

    CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
    for cls_i, (sv, cls_name) in enumerate(zip(shap_np, CLASS_NAMES)):
        ax = fig.add_subplot(4, 1, cls_i + 1)
        # Aggregate abs SHAP across colour channels for a single-channel heatmap
        sv_agg = np.abs(sv).mean(axis=-1)   # [N, H, W]
        sv_agg_norm = (sv_agg - sv_agg.min()) / (sv_agg.max() - sv_agg.min() + 1e-8)

        # Tile all n_explain images side by side
        strip = np.concatenate(list(sv_agg_norm), axis=1)  # [H, N*W]
        ax.imshow(strip, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"SHAP attributions — class: {cls_name}", fontweight="bold")
        ax.axis("off")

    plt.suptitle("SHAP DeepExplainer Summary (|attribution| per class)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    """Load checkpoint, run evaluation, generate confusion matrix and SHAP plot."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    class_names = test_loader.dataset.classes   # type: ignore[attr-defined]

    # ── Load model ────────────────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint.get("model_name", args.model)

    if model_name == "cnn":
        model = RetinalCNN(num_classes=4)
    else:
        model = HybridModel(num_classes=4)

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    print(f"Loaded {model_name.upper()} from epoch {checkpoint.get('epoch', '?')} "
          f"(val acc: {checkpoint.get('val_acc', 0) * 100:.2f}%)")

    # ── Inference ─────────────────────────────────────────────────────────────
    preds, labels = run_inference(model, test_loader, device)
    accuracy = (preds == labels).mean()
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    # ── Per-class report ──────────────────────────────────────────────────────
    print_classification_report(preds, labels, class_names)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm_path = os.path.join(args.results_dir, "confusion_matrix.png")
    plot_confusion_matrix(preds, labels, class_names, save_path=cm_path)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if not args.skip_shap:
        shap_path = os.path.join(args.results_dir, "shap_summary.png")
        run_shap_explanations(
            model, test_loader, device,
            n_background=args.shap_background,
            n_explain=args.shap_samples,
            save_path=shap_path,
        )
    else:
        print("SHAP skipped (--skip_shap flag set).")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RetinalCNN or HybridModel."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to the .pth checkpoint file saved by train.py."
    )
    parser.add_argument(
        "--model", type=str, default="cnn", choices=["cnn", "hybrid"],
        help="Model type (fallback if checkpoint does not store model_name)."
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Root directory containing 'train/' and 'test/' sub-folders."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
    )
    parser.add_argument(
        "--skip_shap", action="store_true",
        help="Skip SHAP computation (fast evaluation mode)."
    )
    parser.add_argument(
        "--shap_background", type=int, default=50,
        help="Number of background samples for SHAP DeepExplainer."
    )
    parser.add_argument(
        "--shap_samples", type=int, default=20,
        help="Number of test samples to explain with SHAP."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
