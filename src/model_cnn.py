"""
model_cnn.py — Custom CNN for Retinal OCT Classification.

Architecture overview
─────────────────────
    Input  [B, 3, 224, 224]
      │
      ├─ Block 1 ─ Conv(3→32) ─ BN ─ ReLU ─ MaxPool  → [B,  32, 112, 112]
      ├─ Block 2 ─ Conv(32→64) ─ BN ─ ReLU ─ MaxPool → [B,  64,  56,  56]
      ├─ Block 3 ─ Conv(64→128)─ BN ─ ReLU ─ MaxPool → [B, 128,  28,  28]
      │
      ├─ GlobalAveragePooling                          → [B, 128]
      └─ Linear(128 → 4)                              → [B,   4]

Why Global Average Pooling instead of Flatten?
──────────────────────────────────────────────
After Block 3 the spatial tensor is [B, 128, 28, 28].

• Flatten → [B, 128 × 28 × 28] = [B, 100352].  This destroys every spatial
  relationship between feature map cells.  Grad-CAM requires that the gradient
  ∂y^c/∂A^k_ij is computed with respect to individual spatial positions (i, j)
  in activation map A^k.  After Flatten those positions are merged into a single
  long vector, making it impossible to form a 2-D heatmap without reverse
  engineering the pixel offsets.

• GAP → [B, 128].  Each of the 128 output neurons is the *spatial mean* of one
  entire feature map.  The intermediate activation A^k (shape [28, 28]) is still
  intact as a named layer output, so PyTorch hooks can capture it and its
  gradient during the backward pass.  This is what makes Grad-CAM work.

  Additionally, GAP acts as a structural regulariser: it forces each filter to
  produce a meaningful global summary rather than memorising spatial locations,
  which improves generalisation on small medical datasets.

Design follows:
    Zhou et al. (2016). Learning Deep Features for Discriminative Localisation.
    CVPR.  (Original GAP + CAM paper.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
#  Building block: a single convolutional stage
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    One convolutional block: Conv2d → BatchNorm2d → ReLU → MaxPool2d.

    Args:
        in_channels:  Number of input feature maps (channels).
        out_channels: Number of output feature maps produced by this block.
        kernel_size:  Spatial extent of each convolutional filter.  Default 3.
        pool_size:    Spatial extent of the max-pooling window.  Default 2.

    Role of each sub-layer:
        Conv2d      — Learns local spatial patterns (edges, textures, fluid
                       reflectance artefacts in OCT).
        BatchNorm2d — Normalises activations across the mini-batch, which:
                       (a) reduces internal covariate shift,
                       (b) allows higher learning rates without divergence,
                       (c) provides slight L2-like regularisation.
        ReLU        — Non-linearity; sets negative activations to zero,
                       producing sparse, interpretable feature maps.
        MaxPool2d   — Halves spatial resolution, forcing the network to learn
                       translation-invariant representations and expanding the
                       receptive field of later layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        # padding=1 keeps spatial dimensions unchanged *before* pooling, so
        # output size = floor(H / pool_size) — predictable and clean.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,   # BN already has a learnable bias term (β)
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Output tensor of shape [B, out_channels, H // pool_size, W // pool_size].
        """
        return self.pool(self.relu(self.bn(self.conv(x))))


# ──────────────────────────────────────────────────────────────────────────────
#  Full CNN model
# ──────────────────────────────────────────────────────────────────────────────

class RetinalCNN(nn.Module):
    """
    Lightweight CNN for 4-class retinal OCT classification.

    The architecture is intentionally shallow (3 conv blocks) so that:
      1. It trains to convergence on OCT in ≤ 20 epochs on a single GPU.
      2. The Grad-CAM heatmaps are not too diffuse — deeper networks spread
         attention across many fine-grained feature maps, making heatmaps harder
         to interpret clinically.

    Args:
        num_classes: Number of output classes.  Default 4 (CNV, DME, DRUSEN, NORMAL).
        in_channels: Number of image channels.  Default 3 (RGB).

    Layer filter progression: 3 → 32 → 64 → 128.
    Doubling channels at each stage is empirically effective (cf. VGG design).
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 3) -> None:
        super().__init__()

        # ── Convolutional backbone ────────────────────────────────────────────
        # Block 1: low-level feature extraction (edges, brightness gradients)
        # Input [B, 3, 224, 224] → Output [B, 32, 112, 112]
        self.block1 = ConvBlock(in_channels, 32)

        # Block 2: mid-level features (curved structures, layer boundaries)
        # Input [B, 32, 112, 112] → Output [B, 64, 56, 56]
        self.block2 = ConvBlock(32, 64)

        # Block 3: high-level semantic features (fluid pockets, disrupted layers)
        # Input [B, 64, 56, 56] → Output [B, 128, 28, 28]
        # This is the last conv layer — GradCAM hooks attach here.
        self.block3 = ConvBlock(64, 128)

        # ── Global Average Pooling ────────────────────────────────────────────
        # AdaptiveAvgPool2d(1) computes the mean over the entire H×W spatial
        # grid for each of the 128 channels, yielding a [B, 128, 1, 1] tensor.
        # The subsequent view(-1) squeezes it to [B, 128].
        # Crucially: *before* GAP, the activation map A^k is still [28, 28],
        # which is exactly what Grad-CAM needs.
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ── Classifier head ───────────────────────────────────────────────────
        # A single linear layer maps the 128-dim pooled representation to class
        # logits.  The simplicity here is intentional — it means the backbone
        # must carry all the discriminative information, keeping Grad-CAM
        # attributions directly tied to convolutional features.
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Full forward pass.

        Args:
            x: Input image batch of shape [B, 3, 224, 224].
        Returns:
            Class logits of shape [B, num_classes].
        """
        x = self.block1(x)    # [B,  32, 112, 112]
        x = self.block2(x)    # [B,  64,  56,  56]
        x = self.block3(x)    # [B, 128,  28,  28]  ← Grad-CAM target
        x = self.gap(x)       # [B, 128,   1,   1]
        x = x.view(x.size(0), -1)   # [B, 128]
        x = self.classifier(x)      # [B,   4]
        return x

    def get_last_conv_layer(self) -> nn.Module:
        """
        Return the final convolutional layer (Conv2d inside block3).

        Grad-CAM hooks must attach to the *last* convolutional layer because:
        • It has the deepest (most semantically rich) feature representations.
        • Its activations A^k still retain 2-D spatial structure [28, 28],
          which is upsampled back to the input resolution to create the heatmap.
        • Hooking an earlier layer would produce heatmaps that respond to
          lower-level cues (edges) rather than pathology-specific patterns.

        Returns:
            The nn.Conv2d module inside self.block3.
        """
        # block3 → ConvBlock → .conv is the nn.Conv2d
        return self.block3.conv

    def feature_maps(self, x: Tensor) -> Tensor:
        """
        Return the activation maps from the last conv block (before GAP).

        Useful for debugging, visualisation, and the Grad-CAM overlay step.

        Args:
            x: Input tensor [B, 3, 224, 224].
        Returns:
            Activation tensor [B, 128, 28, 28].
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


# ──────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = RetinalCNN(num_classes=4)
    dummy = torch.randn(4, 3, 224, 224)   # batch of 4 images
    logits = model(dummy)
    print(f"Output shape   : {logits.shape}")    # expect [4, 4]
    print(f"Last conv layer: {model.get_last_conv_layer()}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")     # expect ~160k
