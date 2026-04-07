"""
gradcam.py — Grad-CAM Implementation from Scratch using PyTorch Hooks.

──────────────────────────────────────────────────────────────────────────────
  THE MATH
──────────────────────────────────────────────────────────────────────────────

  Grad-CAM (Selvaraju et al., 2017. ICCV) produces a coarse localisation map
  highlighting which regions of the input image are most relevant to a model's
  prediction for a given class c.

  Let:
    A^k        — activation map of the k-th feature map in the final conv layer.
                  Shape: [K, H, W] where K = number of filters, H×W = spatial dims.
    y^c        — the raw (pre-softmax) score for class c.
    Z          — total number of spatial positions = H × W.

  Step 1 — Gradient-based importance weights:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │      α_k^c  =  (1/Z)  Σ_i  Σ_j   ∂y^c / ∂A^k_{ij}                   │
  │                                                                         │
  │  α_k^c is the mean gradient of the class score y^c with respect to     │
  │  every spatial position (i, j) of the k-th activation map.             │
  │  It answers: "How much does tweaking filter k, averaged across all     │
  │  positions, change the score for class c?"                              │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  Step 2 — Weighted combination of activation maps:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │      L^c_{GradCAM}  =  ReLU ( Σ_k  α_k^c · A^k )                     │
  │                                                                         │
  │  Each activation map A^k is scaled by its importance weight α_k^c and  │
  │  summed.  The ReLU keeps only features that have a positive effect on   │
  │  the target class (suppresses regions that push the score *down*).      │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  Step 3 — Upsampling:
      L^c is typically of size 7×7 or 14×14 (depending on network depth).
      Bilinear interpolation resizes it to the input image dimensions (224×224)
      so the heatmap aligns pixel-for-pixel with the original scan.

  Reference:
      Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep
      Networks via Gradient-based Localisation. ICCV.
      https://arxiv.org/abs/1610.02391

──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping — from scratch.

    Registers forward and backward hooks on a specified convolutional layer to
    capture:
        • The forward activation maps  A^k   (what the layer *sees*)
        • The gradients               ∂y^c/∂A^k  (how important each location is)

    Then combines them according to the Grad-CAM formula to produce a heatmap.

    Args:
        model:        A PyTorch model (e.g. RetinalCNN or HybridModel).
        target_layer: The nn.Module whose activations and gradients we capture.
                      Use model.get_last_conv_layer() for a sensible default.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        # Containers filled by the hooks during forward / backward passes
        self._activations: Optional[Tensor] = None   # A^k  shape [1, K, H, W]
        self._gradients:   Optional[Tensor] = None   # ∂y^c/∂A^k same shape

        # ── Register hooks ────────────────────────────────────────────────────
        # A forward hook fires *after* the layer computes its output during the
        # forward pass and receives (module, input, output).
        self._forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )
        # A backward hook fires during the backward pass and receives
        # (module, grad_input, grad_output).  grad_output[0] is ∂L/∂A^k.
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    # ── Hook callbacks ────────────────────────────────────────────────────────

    def _save_activations(self, module, input, output) -> None:
        """
        Forward hook: store the feature maps produced by target_layer.

        Args:
            module: The layer being hooked (target_layer).
            input:  Inputs to target_layer (unused here).
            output: Output of target_layer — this IS A^k, shape [B, K, H, W].
        """
        # Detach from the graph to save memory; we only need the values, not
        # the gradient through the storage itself.
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        """
        Backward hook: store the gradients flowing back through target_layer.

        Args:
            module:      The layer being hooked.
            grad_input:  Gradients w.r.t. the *inputs* of target_layer (unused).
            grad_output: Tuple of gradients w.r.t. the *outputs* of target_layer.
                         grad_output[0] has shape [B, K, H, W] — same as A^k.
        """
        # grad_output[0] = ∂L/∂A^k — exactly what we need for the formula
        self._gradients = grad_output[0].detach()

    # ── Core Grad-CAM computation ─────────────────────────────────────────────

    def generate(
        self,
        input_tensor: Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Run the forward pass, compute gradients, and return the Grad-CAM heatmap.

        Steps executed:
            1. Zero any existing gradients.
            2. Forward pass — hooks populate self._activations.
            3. Identify target class (argmax if class_idx is None).
            4. Create a one-hot score vector and backpropagate — hooks populate
               self._gradients.
            5. Compute α_k^c = mean(∂y^c/∂A^k) over spatial dimensions.
            6. Weighted sum: L = Σ_k α_k^c · A^k.
            7. ReLU(L) — drop negative contributions.
            8. Normalise to [0, 1].

        Args:
            input_tensor: Pre-processed image tensor of shape [1, 3, H, W].
                          Must have requires_grad=False (we set it internally).
            class_idx:    Target class index for which to generate the heatmap.
                          If None, uses the model's top predicted class.

        Returns:
            heatmap: numpy array of shape [H_input, W_input] with values in [0, 1].
                     Spatial size matches the *input* image resolution (after
                     bilinear upsampling from the activation map size).
        """
        # Reset stored state so stale activations/gradients from a previous
        # call never silently contaminate the next one.
        self._activations = None
        self._gradients   = None

        self.model.eval()

        # Ensure gradients can flow through the input tensor
        input_tensor = input_tensor.clone().requires_grad_(True)

        # ── Step 1: Forward pass ─────────────────────────────────────────────
        # The forward hook fires here and stores A^k
        output = self.model(input_tensor)   # [1, num_classes]

        # ── Step 2: Identify target class ────────────────────────────────────
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        # ── Step 3: Backpropagate through the target class score ──────────────
        # Zero previous gradients so accumulated history does not contaminate
        self.model.zero_grad()

        # Create a one-hot score: only the target class logit contributes to loss.
        # Unlike using CrossEntropyLoss, this lets us compute the gradient of
        # *exactly* y^c (not the log-probability), matching the Grad-CAM paper.
        score = output[0, class_idx]
        score.backward()
        # The backward hook fires here and stores ∂y^c/∂A^k

        # ── Step 4: Compute importance weights α_k^c ─────────────────────────
        #                                                                        #
        #   α_k^c = (1 / Z) Σ_i Σ_j  ∂y^c / ∂A^k_{ij}                       #
        #                                                                        #
        # self._gradients shape: [1, K, H_feat, W_feat]
        # Mean over spatial dims (dim=2, dim=3) → α shape: [1, K, 1, 1]
        assert self._gradients is not None, "Backward hook did not fire — check hooks."
        assert self._activations is not None, "Forward hook did not fire — check hooks."

        alpha = self._gradients.mean(dim=(2, 3), keepdim=True)   # [1, K, 1, 1]

        # ── Step 5: Weighted combination ──────────────────────────────────────
        #                                                                        #
        #   L^c_{GradCAM} = ReLU ( Σ_k  α_k^c · A^k )                        #
        #                                                                        #
        # Broadcast: alpha [1, K, 1, 1] × activations [1, K, H, W]
        weighted_maps = alpha * self._activations   # [1, K, H_feat, W_feat]
        cam = weighted_maps.sum(dim=1, keepdim=True)   # [1, 1, H_feat, W_feat]

        # ── Step 6: ReLU — keep only positive influences ─────────────────────
        cam = F.relu(cam)

        # ── Step 7: Upsample to input image spatial resolution ────────────────
        # Bilinear interpolation provides smooth heatmaps without block artefacts.
        H_in, W_in = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(
            cam,
            size=(H_in, W_in),
            mode="bilinear",
            align_corners=False,
        )   # [1, 1, H_in, W_in]

        # ── Step 8: Normalise to [0, 1] ───────────────────────────────────────
        cam = cam.squeeze().cpu().numpy()   # [H_in, W_in]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam   # shape [H_in, W_in], values in [0, 1]

    # ── Visualisation helpers ─────────────────────────────────────────────────

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: int = 2,  # cv2.COLORMAP_JET = 2; avoids referencing cv2 at class-definition time
    ) -> np.ndarray:
        """
        Blend the Grad-CAM heatmap over the original image.

        Args:
            heatmap:        Float array [H, W] in [0, 1] from self.generate().
            original_image: uint8 RGB array [H, W, 3] (the raw image, unnormalized).
            alpha:          Opacity of the heatmap overlay.  Default 0.4.
            colormap:       OpenCV colormap for the heatmap.  Default COLORMAP_JET.

        Returns:
            Blended uint8 RGB image [H, W, 3].
        """
        if not HAS_CV2:
            raise ImportError(
                "opencv-python is required for overlay_heatmap.  "
                "Install it with: pip install opencv-python"
            )

        # Convert heatmap: float[0,1] → uint8[0,255] → BGR colormap → RGB
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_bgr   = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_rgb   = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Ensure original_image is uint8 RGB of matching size
        if original_image.dtype != np.uint8:
            original_image = np.uint8(255 * original_image)

        # Alpha blend: overlay = α·heatmap + (1-α)·original
        overlay = cv2.addWeighted(heatmap_rgb, alpha, original_image, 1 - alpha, 0)
        return overlay

    def remove_hooks(self) -> None:
        """
        Remove the registered hooks.

        Call this after finishing all Grad-CAM computations to prevent memory
        leaks from hooks storing large tensors indefinitely.
        """
        self._forward_hook.remove()
        self._backward_hook.remove()

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()


# ──────────────────────────────────────────────────────────────────────────────
#  Grid visualisation utility
# ──────────────────────────────────────────────────────────────────────────────

def _denormalize(tensor: Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and convert to uint8 numpy array.

    Args:
        tensor: Normalised image tensor [3, H, W].
    Returns:
        uint8 numpy array [H, W, 3].
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy()   # [H, W, 3]
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    return np.uint8(255 * img)


def generate_gradcam_grid(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    images: List[Tensor],       # one image tensor per class [1, 3, H, W]
    class_names: List[str],
    save_path: str = "results/gradcam_grid.png",
    device: str = "cpu",
) -> None:
    """
    Produce the 2×N Grad-CAM grid described in the task specification.

    Layout:
        Row 0: original images (one per class)
        Row 1: Grad-CAM overlays for the model's top-predicted class

    Args:
        model:        Trained model in eval mode.
        target_layer: Last conv layer (via model.get_last_conv_layer()).
        images:       List of image tensors, one per class, shape [1, 3, H, W].
        class_names:  Class name strings (e.g. ['CNV', 'DME', 'DRUSEN', 'NORMAL']).
        save_path:    Where to write the PNG file.
        device:       'cuda' or 'cpu'.

    Returns:
        None (saves the figure to disk and displays it).
    """
    import os
    save_dir = os.path.dirname(save_path)
    if save_dir:  # os.path.dirname returns '' for bare filenames — makedirs('') raises FileNotFoundError
        os.makedirs(save_dir, exist_ok=True)

    model.to(device).eval()
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, n_classes, figsize=(4 * n_classes, 8))
    fig.suptitle("Grad-CAM Visualisations by Class", fontsize=15, fontweight="bold")

    with GradCAM(model, target_layer) as gcam:
        for col, (img_tensor, cls_name) in enumerate(zip(images, class_names)):
            img_tensor = img_tensor.to(device)

            # Raw RGB image for display
            raw_img = _denormalize(img_tensor.squeeze(0))

            # Generate heatmap
            heatmap = gcam.generate(img_tensor)
            overlay = gcam.overlay_heatmap(heatmap, raw_img.copy())

            # Row 0 — original
            axes[0, col].imshow(raw_img)
            axes[0, col].set_title(cls_name, fontsize=12, fontweight="bold")
            axes[0, col].axis("off")

            # Row 1 — Grad-CAM overlay
            axes[1, col].imshow(overlay)
            axes[1, col].set_title("Grad-CAM", fontsize=10)
            axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grad-CAM grid saved to {save_path}")
