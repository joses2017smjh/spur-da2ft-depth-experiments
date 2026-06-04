"""
_mvp_precompiled_decompiled.py
==============================
RECONSTRUCTED SOURCE — recovered by disassembling _mvp_precompiled.pyc
(Python 3.10 bytecode) and rebuilding the source line-by-line from the
bytecode + the original docstrings embedded in the .pyc.

This is the RGB+D encoder the production model actually uses:
    MVP_MODEL/mvp_stereo_dino_model.py:400  self.encoder = DINOv2ViTLEncoder()

Original file (per code-object metadata): mvp_depth_refine_model.py
Only the two classes relevant to "where is the RGB-depth skip fusion"
are reconstructed here: DepthSideBranch and DINOv2ViTLEncoder.
Logic verified against the disassembly; docstrings are verbatim from the .pyc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

_DINO_DIM = 1024   # ViT-L token dimension
_SKIP_CH  = 256    # projected skip channel size
_BOTT_CH  = 512    # bottleneck channel size


class DepthSideBranch(nn.Module):
    """Small CNN that extracts multi-scale features from the PRO depth map.

    DINOv2 only accepts RGB, so depth is processed here in a separate branch
    and fused with ViT features at the skip-connection level.

    WHY 3 LAYERS?
    PRO depth is already a processed, meaningful signal — not raw sensor noise.
    Three strided convolutions are enough to produce features at the 3 spatial
    scales (H, H/2, H/4) where depth structure is most useful to fuse.
    Deeper processing would overfit on the small dataset.

    Outputs (3 spatial scales, matched to ViT skip targets):
        f0: (B,  32, H,    W  )  — full-res depth gradients / edges
        f1: (B,  64, H/2,  W/2)  — mid-res depth shape
        f2: (B, 128, H/4,  W/4)  — coarse depth regions
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            d: (B, 1, H, W) — PRO depth map for one view

        Returns:
            f0: (B,  32, H,    W  )
            f1: (B,  64, H/2,  W/2)
            f2: (B, 128, H/4,  W/4)
        """
        f0 = self.layer1(d)
        f1 = self.layer2(f0)
        f2 = self.layer3(f1)
        return f0, f1, f2


class DINOv2ViTLEncoder(nn.Module):
    """DINOv2 ViT-L encoder (frozen) + trainable depth side branch fusion.

    Replaces UNetEncoder. Produces the same interface: bottleneck + 4 skips.
    The spatial resolution of the output depth map is NOT affected by the
    larger channel sizes — the decoder's Up blocks use F.interpolate to
    upsample spatial dims independently of channel count.

    ┌──────────────────────────────────────────────────────────────────┐
    │  RGB (3ch)  ──► DINOv2 ViT-L (FROZEN)                              │
    │                 │  extract tokens at blocks 5,11,17,23             │
    │                 │  reshape (B,N,1024) → (B,1024,h_tok,w_tok)       │
    │                 │  project 1024 → 256 (skips) / 512 (bottleneck)   │
    │                 │  upsample to target skip resolutions             │
    │                                                                    │
    │  depth (1ch) ──► DepthSideBranch (TRAINABLE)                       │
    │                  features at H, H/2, H/4                           │
    │                                                                    │
    │  FUSION (TRAINABLE): cat(ViT_skip, depth_feat) → 1×1 conv          │
    │    s0: 256+32  → 256  at (H,   W  )                                │
    │    s1: 256+64  → 256  at (H/2, W/2)                                │
    │    s2: 256+128 → 256  at (H/4, W/4)                                │
    │    s3: 256           at (H/8, W/8)  ← no depth fusion (too coarse) │
    │    b:  512           at (H/14,W/14) ← bottleneck                   │
    └──────────────────────────────────────────────────────────────────┘

    out_ch = 512 so fuse_conv in MVDINOv2PoseConcat is identical to
    the original: 3×(512+32) → 512.
    """

    _HOOK_BLOCKS = (5, 11, 17, 23)   # block indices to extract (0-indexed)

    def __init__(self) -> None:
        super().__init__()
        # Frozen DINOv2 ViT-L backbone
        self.dino = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True
        )
        for p in self.dino.parameters():
            p.requires_grad = False

        # 1×1 projections: 1024 ViT channels → skip / bottleneck channels
        self.proj_l6  = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_l12 = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_l18 = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_l24 = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_b   = nn.Conv2d(_DINO_DIM, _BOTT_CH, 1)

        # Trainable depth side branch
        self.depth_branch = DepthSideBranch()

        # Trainable fusion: concat(ViT skip, depth feat) → 1×1 conv → 256
        self.fuse_s0 = nn.Conv2d(_SKIP_CH + 32,  _SKIP_CH, 1)
        self.fuse_s1 = nn.Conv2d(_SKIP_CH + 64,  _SKIP_CH, 1)
        self.fuse_s2 = nn.Conv2d(_SKIP_CH + 128, _SKIP_CH, 1)

        self.out_ch = _BOTT_CH

    @staticmethod
    def _tokens_to_spatial(
        tokens: torch.Tensor, h_tok: int, w_tok: int
    ) -> torch.Tensor:
        """Reshape flat token sequence → 2D spatial map.

        DINOv2 pads images to the nearest multiple of patch_size (14) before
        tokenising, so w_tok = ceil(W/14), h_tok = ceil(H/14).

        Args:
            tokens: (B, N, 1024)  N = h_tok * w_tok
            h_tok:  token grid height
            w_tok:  token grid width
        Returns:
            (B, 1024, h_tok, w_tok)
        """
        B, N, C = tokens.shape
        return tokens.permute(0, 2, 1).reshape(B, C, h_tok, w_tok)

    def forward(
        self, rgb: torch.Tensor, depth: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            rgb:   (B, 3, H, W) — ImageNet-normalised RGB for one view
            depth: (B, 1, H, W) — PRO depth for one view

        Returns:
            b:     (B, 512, H/14, W/14) — bottleneck (spatial size uses ceil)
            skips: (s0, s1, s2, s3) — all (B, 256, *)
                   s0 at (H, W), s1 at (H/2, W/2),
                   s2 at (H/4, W/4), s3 at (H/8, W/8)
        """
        B, _, H, W = rgb.shape

        # Pad RGB to a multiple of patch_size=14
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        rgb_in = F.pad(rgb, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else rgb
        h_tok = (H + pad_h) // 14
        w_tok = (W + pad_w) // 14

        # ---- RGB stream: frozen DINOv2 features at 4 block depths ----
        raw = self.dino.get_intermediate_layers(
            rgb_in, n=self._HOOK_BLOCKS, return_class_token=False
        )
        f6, f12, f18, f24 = [
            self._tokens_to_spatial(t, h_tok, w_tok) for t in raw
        ]

        # Project 1024 → 256 / 512
        f6  = self.proj_l6(f6)
        f12 = self.proj_l12(f12)
        f18 = self.proj_l18(f18)
        s3  = self.proj_l24(f24)   # deepest skip — no depth fusion
        b   = self.proj_b(f24)     # bottleneck, stays at (h_tok, w_tok)

        # Upsample ViT skips to UNet skip resolutions
        s0_vit = F.interpolate(f6,  size=(H,      W     ), mode='bilinear', align_corners=False)
        s1_vit = F.interpolate(f12, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        s2_vit = F.interpolate(f18, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        s3     = F.interpolate(s3,  size=(H // 8, W // 8), mode='bilinear', align_corners=False)

        # ---- Depth stream: trainable side branch ----
        d0, d1, d2 = self.depth_branch(depth)

        # ---- RGB-DEPTH SKIP FUSION: concat then 1×1 conv ----
        s0 = self.fuse_s0(torch.cat([s0_vit, d0], dim=1))
        s1 = self.fuse_s1(torch.cat([s1_vit, d1], dim=1))
        s2 = self.fuse_s2(torch.cat([s2_vit, d2], dim=1))
        # s3 has no depth fusion (too coarse)

        return b, (s0, s1, s2, s3)


# =============================================================================
# Decoder building blocks (recovered from the same .pyc)
# =============================================================================

class ConvBlock(nn.Module):
    """Two 3×3 convolution layers with BatchNorm and ReLU.

    Pattern: Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU. Each conv preserves
    spatial dims (padding=1). bias=False because BatchNorm has its own bias.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Decoder upsampling step: double resolution, concatenate skip, ConvBlock.

      1. Bilinear upsample x to the skip connection's spatial size
      2. Concatenate with skip along the channel dim
      3. ConvBlock to process the combined features
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DINODecoder(nn.Module):
    """Decoder matched to DINOv2ViTLEncoder skip channel sizes.

    Identical logic to UNetDecoder but uses uniform _SKIP_CH=256 at all
    skip levels (instead of the 64/128/256/512 progression of the CNN path).

    Channel flow:
      b      (B, 512, H/14, W/14)  ← fused bottleneck
      up3  + s3(256) → (B, 256, H/8,  W/8)
      up2  + s2(256) → (B, 256, H/4,  W/4)
      up1  + s1(256) → (B, 128, H/2,  W/2)
      up0  + s0(256) → (B,  64, H,    W  )
      head            → (B,   1, H,    W  )  ← depth prediction
    """

    def __init__(self) -> None:
        super().__init__()
        self.up3  = Up(_BOTT_CH, _SKIP_CH, 256)   # 512 + 256 → 256
        self.up2  = Up(256,      _SKIP_CH, 256)   # 256 + 256 → 256
        self.up1  = Up(256,      _SKIP_CH, 128)   # 256 + 256 → 128
        self.up0  = Up(128,      _SKIP_CH,  64)   # 128 + 256 → 64
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(
        self, b: torch.Tensor, skips: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        Args:
            b:     (B, 512, H/14, W/14) — fused multi-view bottleneck
            skips: (s0, s1, s2, s3) — per-view skip connections

        Returns:
            (B, 1, H, W) — refined depth for one view
        """
        s0, s1, s2, s3 = skips
        x = self.up3(b, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)
