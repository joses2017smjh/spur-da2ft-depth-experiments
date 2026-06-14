"""
simple_stereo/model.py
======================
Minimal stereo depth refinement — PRO depth input only, no RGB, no pose.

Two model variants sharing the same interface:

  StereoDepthCNN   — from-scratch UNet encoder/decoder
  StereoDepthDINO  — frozen DINOv2 ViT-L encoder, depth expanded to 3ch

Both accept:
  d_pro  : (B, 2, 1, H, W)  — stereo PRO depth pair (left + right)
  no_fusion flag:
    False (default) — cross-view fusion: both views share a fused bottleneck
    True            — each view decoded independently (monocular baseline)

Both output:
  d_pred : (B, 2, 1, H, W)  — refined depth per view

Loss & metrics at the bottom (SiLog training loss, RMSE eval metric).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# =============================================================================
# Shared building blocks
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# =============================================================================
# CNN path — encoder / decoder
# =============================================================================

class DepthEncoder(nn.Module):
    """UNet encoder for a single-channel (depth-only) input.

    Channel progression (base=64):
      s0: (B,  64, H,    W  )
      s1: (B, 128, H/2,  W/2)
      s2: (B, 256, H/4,  W/4)
      s3: (B, 512, H/8,  W/8)
      b:  (B, 512, H/16, W/16)
    """

    def __init__(self, base: int = 64) -> None:
        super().__init__()
        self.enc0 = ConvBlock(1,        base)
        self.enc1 = Down(base,      base * 2)
        self.enc2 = Down(base * 2,  base * 4)
        self.enc3 = Down(base * 4,  base * 8)
        self.bot  = Down(base * 8,  base * 8)
        self.out_ch = base * 8

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b  = self.bot(s3)
        return b, (s0, s1, s2, s3)


class DepthDecoder(nn.Module):
    """UNet decoder matching DepthEncoder skip channel sizes."""

    def __init__(self, base: int = 64) -> None:
        super().__init__()
        Cb = base * 8
        self.up3 = Up(Cb,       base * 8, base * 8)
        self.up2 = Up(base * 8, base * 4, base * 4)
        self.up1 = Up(base * 4, base * 2, base * 2)
        self.up0 = Up(base * 2, base,     base)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

    def forward(
        self, b: torch.Tensor, skips: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        s0, s1, s2, s3 = skips
        x = self.up3(b,  s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)


# =============================================================================
# Model 1: StereoDepthCNN
# =============================================================================

class StereoDepthCNN(nn.Module):
    """Stereo depth refinement — from-scratch UNet, depth-only input.

    With fusion (default):
        Both bottlenecks concatenated → 1×1 conv → shared fused bottleneck.
        Views can exchange depth information at the most compressed level.

    Without fusion (no_fusion=True):
        Each view decoded with its own bottleneck — two independent monocular
        refinement runs sharing only weights, not information.
    """

    def __init__(self, base: int = 64, no_fusion: bool = False) -> None:
        super().__init__()
        self.no_fusion = no_fusion
        self.encoder = DepthEncoder(base=base)
        Cb = self.encoder.out_ch
        if not no_fusion:
            self.fuse_conv = nn.Conv2d(2 * Cb, Cb, kernel_size=1, bias=True)
        self.decoder = DepthDecoder(base=base)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, d_pro: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d_pro: (B, 2, 1, H, W) — stereo PRO depth pair

        Returns:
            (B, 2, 1, H, W) — refined depth per view
        """
        B, V = d_pro.shape[:2]

        bottlenecks: List[torch.Tensor] = []
        all_skips: List[Tuple] = []
        for i in range(V):
            b_i, skips_i = self.encoder(d_pro[:, i])
            bottlenecks.append(b_i)
            all_skips.append(skips_i)

        if self.no_fusion:
            b_list = bottlenecks
        else:
            b_fused = self.fuse_conv(torch.cat(bottlenecks, dim=1))
            b_list = [b_fused, b_fused]

        preds = [self.decoder(b_list[i], all_skips[i]) for i in range(V)]
        return torch.stack(preds, dim=1)   # (B, 2, 1, H, W)


# =============================================================================
# DINO path — encoder / decoder
# =============================================================================

_DINO_DIM = 1024   # ViT-L token dimension
_SKIP_CH  = 256    # projected skip channel size
_BOTT_CH  = 512    # bottleneck channel size


class DINODepthEncoder(nn.Module):
    """Frozen DINOv2 ViT-L encoder for depth-only input.

    Depth is expanded from 1 → 3 channels before DINOv2 (pseudo-RGB).
    ViT-L intermediate features at 4 block depths are projected to uniform
    _SKIP_CH=256 and upsampled to standard UNet skip resolutions.

    Returns same interface as DepthEncoder: (bottleneck, (s0,s1,s2,s3)).
    """

    _HOOK_BLOCKS = [5, 11, 17, 23]   # block indices to extract (0-indexed)

    def __init__(self) -> None:
        super().__init__()
        self.dino = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitl14', pretrained=True
        )
        for p in self.dino.parameters():
            p.requires_grad = False

        self.proj_s0 = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_s1 = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_s2 = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_s3 = nn.Conv2d(_DINO_DIM, _SKIP_CH, 1)
        self.proj_b  = nn.Conv2d(_DINO_DIM, _BOTT_CH, 1)
        self.out_ch  = _BOTT_CH

    @staticmethod
    def _to_spatial(
        tokens: torch.Tensor, h_tok: int, w_tok: int
    ) -> torch.Tensor:
        B, N, C = tokens.shape
        return tokens.permute(0, 2, 1).reshape(B, C, h_tok, w_tok)

    def forward(
        self, depth: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Args:
            depth: (B, 1, H, W) — single-view PRO depth

        Returns:
            b:     (B, 512, h_tok, w_tok)  bottleneck (~H/14)
            skips: (s0, s1, s2, s3) each (B, 256, *)
        """
        B, _, H, W = depth.shape

        # Expand depth → 3 channels for DINOv2 (trained on RGB)
        rgb_in = depth.expand(-1, 3, -1, -1)   # (B, 3, H, W), no copy

        # Pad to multiples of patch_size=14
        pad_h = (14 - H % 14) % 14
        pad_w = (14 - W % 14) % 14
        if pad_h or pad_w:
            rgb_in = F.pad(rgb_in, (0, pad_w, 0, pad_h))
        h_tok = (H + pad_h) // 14
        w_tok = (W + pad_w) // 14

        # Extract intermediate tokens at 4 depths
        raw = self.dino.get_intermediate_layers(
            rgb_in, n=self._HOOK_BLOCKS, return_class_token=False
        )
        f6, f12, f18, f24 = [self._to_spatial(t, h_tok, w_tok) for t in raw]

        # Project 1024 → target channel sizes, upsample to skip resolutions
        s0 = F.interpolate(self.proj_s0(f6),  size=(H,     W    ), mode='bilinear', align_corners=False)
        s1 = F.interpolate(self.proj_s1(f12), size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        s2 = F.interpolate(self.proj_s2(f18), size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.proj_s3(f24), size=(H // 8, W // 8), mode='bilinear', align_corners=False)
        b  = self.proj_b(f24)   # stays at (h_tok, w_tok) ≈ H/14

        return b, (s0, s1, s2, s3)


class DINODecoder(nn.Module):
    """Decoder matched to DINODepthEncoder's uniform _SKIP_CH=256 skips."""

    def __init__(self) -> None:
        super().__init__()
        self.up3 = Up(_BOTT_CH, _SKIP_CH, 256)
        self.up2 = Up(256,      _SKIP_CH, 256)
        self.up1 = Up(256,      _SKIP_CH, 128)
        self.up0 = Up(128,      _SKIP_CH,  64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(
        self, b: torch.Tensor, skips: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        s0, s1, s2, s3 = skips
        x = self.up3(b,  s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)
        return self.head(x)


# =============================================================================
# Model 2: StereoDepthDINO
# =============================================================================

class StereoDepthDINO(nn.Module):
    """Stereo depth refinement — frozen DINOv2 ViT-L encoder, depth-only.

    DINOv2 sees depth expanded to 3 channels (pseudo-RGB). The frozen ViT-L
    produces rich multi-scale features; only projection layers + decoder are
    trained (~5M trainable / ~304M frozen).

    With fusion (default):
        Both view bottlenecks fused via 1×1 conv.
    Without fusion (no_fusion=True):
        Each view decoded independently — monocular DINO baseline.
    """

    def __init__(self, no_fusion: bool = False) -> None:
        super().__init__()
        self.no_fusion = no_fusion
        self.encoder = DINODepthEncoder()
        Cb = self.encoder.out_ch   # = 512
        if not no_fusion:
            self.fuse_conv = nn.Conv2d(2 * Cb, Cb, kernel_size=1, bias=True)
        self.decoder = DINODecoder()

    def forward(self, d_pro: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d_pro: (B, 2, 1, H, W) — stereo PRO depth pair

        Returns:
            (B, 2, 1, H, W) — refined depth per view
        """
        B, V = d_pro.shape[:2]

        bottlenecks: List[torch.Tensor] = []
        all_skips: List[Tuple] = []
        for i in range(V):
            b_i, skips_i = self.encoder(d_pro[:, i])
            bottlenecks.append(b_i)
            all_skips.append(skips_i)

        if self.no_fusion:
            b_list = bottlenecks
        else:
            b_fused = self.fuse_conv(torch.cat(bottlenecks, dim=1))
            b_list = [b_fused, b_fused]

        preds = [self.decoder(b_list[i], all_skips[i]) for i in range(V)]
        return torch.stack(preds, dim=1)   # (B, 2, 1, H, W)


# =============================================================================
# Loss & metrics
# =============================================================================

def silog_loss(
    d_pred: torch.Tensor,
    d_gt:   torch.Tensor,
    mask:   torch.Tensor,
    variance_focus: float = 0.85,
    min_depth: float = 0.5,
    max_depth: float = 10.0,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Scale-invariant log loss summed over both views.

    Only trunk-masked pixels with GT depth in [min_depth, max_depth] contribute.

    Returns:
        total: scalar loss for backprop
        per_view: [loss_left, loss_right] for logging
    """
    V = d_pred.shape[1]
    per_view: List[torch.Tensor] = []
    for v in range(V):
        pred = d_pred[:, v].clamp(min=eps)
        gt   = d_gt[:, v].clamp(min=eps)
        m    = (mask[:, v].float()
                * (d_gt[:, v] > min_depth).float()
                * (d_gt[:, v] < max_depth).float())
        n    = m.sum() + eps
        d    = (torch.log(pred) - torch.log(gt)) * m
        var  = (d ** 2).sum() / n - variance_focus * (d.sum() / n) ** 2
        per_view.append(torch.sqrt(var.clamp(min=0.0)))
    return sum(per_view), per_view


def rmse(
    d_pred: torch.Tensor,
    d_gt:   torch.Tensor,
    mask:   torch.Tensor,
    min_depth: float = 0.5,
    max_depth: float = 10.0,
    eps: float = 1e-6,
) -> Tuple[float, List[float]]:
    """Masked RMSE over both views (evaluation only)."""
    V = d_pred.shape[1]
    per_view: List[float] = []
    for v in range(V):
        m = (mask[:, v].float()
             * (d_gt[:, v] > min_depth).float()
             * (d_gt[:, v] < max_depth).float())
        diff2 = (d_pred[:, v] - d_gt[:, v]) ** 2 * m
        per_view.append(torch.sqrt(diff2.sum() / (m.sum() + eps)).item())
    return sum(per_view) / V, per_view
