"""
model_hybrid.py — CNN + Transformer Hybrid for Retinal OCT Classification.

Motivation
──────────
CNNs excel at local feature extraction (edges, textures, fluid pockets in OCT),
but self-attention can capture long-range dependencies — e.g. how a drusen deposit
in the periphery relates to structural changes near the fovea.

This architecture mirrors the design philosophy of:
    Dosovitskiy, A. et al. (2021).  "An Image is Worth 16x16 Words: Transformers for
    Image Recognition at Scale."  ICLR 2021.  arXiv:2010.11929.

    Chen, J. et al. (2021).  "TransUNet: Transformers Make Strong Encoders for Medical
    Image Segmentation."  arXiv:2102.04306.

Here we implement a minimal, principled version:
    1. CNN backbone extracts spatial feature maps  →  [B, 128, 28, 28]
    2. Reshape feature maps into a sequence of spatial tokens
    3. A Transformer encoder applies multi-head self-attention over those tokens
    4. CLS token (or pooled output) feeds a linear classifier

This model is drop-in compatible with train.py and gradcam.py:
    - forward(x) returns logits of shape [B, num_classes]
    - get_last_conv_layer() returns the Conv2d for Grad-CAM hook registration

Architecture:
    Input [B, 3, 224, 224]
      │
      ├─ Shared CNN Backbone (same as RetinalCNN) → [B, 128, 28, 28]
      │
      ├─ Flatten spatial → sequence of tokens [B, 784, 128]
      │    (784 = 28 × 28 spatial positions, each as a 128-dim embedding)
      │
      ├─ Prepend learnable [CLS] token → [B, 785, 128]
      │
      ├─ Add learnable positional embeddings → [B, 785, 128]
      │
      ├─ TransformerEncoder (2 heads, 1 layer, ff_dim=256) → [B, 785, 128]
      │
      ├─ Extract CLS token → [B, 128]
      │
      └─ Linear(128 → 4) → [B, 4]
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor

try:
    from src.model_cnn import ConvBlock
except ModuleNotFoundError:
    from model_cnn import ConvBlock


class HybridModel(nn.Module):
    """
    CNN + Transformer Hybrid for retinal OCT classification.

    The Transformer head is inserted *after* the CNN backbone.  This is
    sometimes called a "late fusion" approach: the CNN shoulders the burden
    of feature extraction, while the Transformer refines the spatial
    relationships between feature positions.

    Args:
        num_classes:    Number of output classes.  Default 4.
        in_channels:    Input image channels.  Default 3.
        d_model:        Dimensionality of each token fed to the Transformer.
                        Must equal the CNN's output channels (128).  Default 128.
        nhead:          Number of self-attention heads.  Default 2.
        num_layers:     Number of stacked TransformerEncoder layers.  Default 1.
        dim_feedforward: Inner size of the Transformer FFN.  Default 256.
        dropout:        Dropout rate inside the Transformer.  Default 0.1.

    Design note on sequence length:
        After block3, feature maps are [B, 128, 28, 28].
        Flattening the spatial dims gives 28 × 28 = 784 tokens.
        Each token is a 128-dim feature vector (one spatial position's embedding).
        784 tokens × 2 attention heads = each head attends over 784 positions.
        This is computationally feasible at batch_size=32 for inference/training.
    """

    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        d_model: int = 128,
        nhead: int = 2,
        num_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Shared CNN Backbone (identical to RetinalCNN) ─────────────────────
        # Using the same architecture ensures fair comparison when both models
        # see the same low/mid/high-level features from the same conv ops.
        self.block1 = ConvBlock(in_channels, 32)    # → [B,  32, 112, 112]
        self.block2 = ConvBlock(32,           64)   # → [B,  64,  56,  56]
        self.block3 = ConvBlock(64,          128)   # → [B, 128,  28,  28]

        # Project CNN output (128 channels) to d_model to support different Transformer sizes
        self.proj = nn.Linear(128, d_model) if d_model != 128 else nn.Identity()
        self.d_model = d_model

        # ── CLS token ─────────────────────────────────────────────────────────
        # A learnable [CLS] token is prepended to the sequence.  After passing
        # through the Transformer encoder, only the CLS token's representation
        # is used for classification — it aggregates global context via attention.
        # (Pattern from BERT / ViT.)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional Embeddings ─────────────────────────────────────────────
        # The Transformer is permutation-invariant by design; without positional
        # information it cannot distinguish "top-left drusen" from "center drusen".
        # We use simple learnable 1-D embeddings (784 spatial positions + 1 CLS).
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 28 * 28, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer Encoder ───────────────────────────────────────────────
        # This follows the hybrid CNN-Transformer design from Dosovitskiy et al. (ViT)
        # and Chen et al. (TransUNet) — a single Transformer encoder layer provides enough
        # cross-position attention without exploding parameter count.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,    # expect [B, seq_len, d_model]
            activation="gelu",   # GELU (Hendrycks & Gimpel, 2016) is standard in vision Transformers
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer norm applied to the CLS token before classification
        # (matches the pre-norm convention from DeiT / ViT)
        self.norm = nn.LayerNorm(d_model)

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Full forward pass: CNN → tokenise → Transformer → classify.

        Args:
            x: Input image batch [B, 3, 224, 224].
        Returns:
            Logits [B, num_classes].
        """
        B = x.size(0)

        # ── CNN backbone ──────────────────────────────────────────────────────
        x = self.block1(x)    # [B,  32, 112, 112]
        x = self.block2(x)    # [B,  64,  56,  56]
        x = self.block3(x)    # [B, 128,  28,  28]  ← Grad-CAM target

        # ── Tokenise: spatial positions become sequence elements ───────────────
        # Flatten (H, W) → S = H × W = 784 and permute to [B, S, C]
        x = x.flatten(2).permute(0, 2, 1)   # [B, 784, 128]
        x = self.proj(x)                    # [B, 784, d_model]

        # ── Prepend CLS token ─────────────────────────────────────────────────
        cls_tokens = self.cls_token.expand(B, -1, -1)   # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)            # [B, 785, d_model]

        # ── Add positional embeddings ─────────────────────────────────────────
        x = x + self.pos_embed   # broadcast over batch

        # ── Transformer encoder ───────────────────────────────────────────────
        x = self.transformer(x)   # [B, 785, d_model]

        # ── Extract CLS token representation ─────────────────────────────────
        cls_out = self.norm(x[:, 0, :])   # [B, d_model]

        # ── Linear classifier ─────────────────────────────────────────────────
        logits = self.classifier(cls_out)   # [B, 4]
        return logits

    def get_last_conv_layer(self) -> nn.Module:
        """
        Return the final convolutional layer (Conv2d inside block3).

        Identical to RetinalCNN.get_last_conv_layer() — this ensures Grad-CAM
        can be applied to the hybrid model without any code changes, enabling
        direct comparison of what the CNN backbone attends to before and after
        the Transformer refines the classification token.

        Returns:
            The nn.Conv2d module inside self.block3.
        """
        return self.block3.conv


# ──────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = HybridModel(num_classes=4)
    dummy = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    print(f"Output shape   : {logits.shape}")    # expect [2, 4]
    print(f"Last conv layer: {model.get_last_conv_layer()}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
