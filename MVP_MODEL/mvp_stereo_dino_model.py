"""
MVP_MODEL/mvp_stereo_dino_model.py
───────────────────────────────────
MVStereoDINOUNet — identical to MVStereoUNet but with the CNN encoder/decoder
replaced by the frozen DINOv2 ViT-L encoder + trainable DINODecoder from
mvp_depth_refine_model.py.

What changes vs MVStereoUNet
─────────────────────────────
  • Encoder:  DINOv2ViTLEncoder (frozen ViT-L backbone + trainable depth side
              branch + skip fusion) instead of from-scratch UNetEncoder.
              forward(rgb, depth) — takes RGB and PRO depth separately because
              DINOv2 only accepts RGB; depth goes through DepthSideBranch.
  • Decoder:  DINODecoder (uniform 256-ch skips) instead of UNetDecoder.
  • ~300 M frozen params (DINOv2 backbone) + ~5 M trainable params.

What stays the same
────────────────────
  • Pose embedding: 3 absolute scalars (delta_z, delta_rot_z, delta_r) →
    PoseProject(in_dim=3, pose_ch=32) → broadcast + cat to bottleneck.
  • Cross-view fusion: 1×1 conv over all n_views (bottleneck + pose).
  • Output activation: softplus (ensures positive depth).
  • n_views parameter: works for 2 (single stereo pair) or 6 (3 pairs).
  • All loss / metric functions imported from mvp_stereo_model.

Losses / metrics (re-exported for convenience)
───────────────────────────────────────────────
  silog_loss_nview, stereo_mv_consistency_loss, masked_rmse_nview
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pose constants and embedding from the shared UNet model file
from mvp_depth_model_Unet import (
    PoseProject,
    _CAM_Z_BASE, _CAM_ROT_Z_BASE, _CAM_R_BASE,
)

# DINOv2 encoder + matching decoder from the 3-view depth refinement model
from mvp_depth_refine_model import (
    DINOv2ViTLEncoder,
    DINODecoder,
)

# Re-export losses / metrics so training scripts only need one import
from mvp_stereo_model import (
    silog_loss_nview,
    stereo_mv_consistency_loss,
    masked_rmse_nview,
)

silog_loss_2view  = silog_loss_nview   # backwards-compat alias
masked_rmse_2view = masked_rmse_nview  # backwards-compat alias

# DepthSideBranch for depth-only model
from mvp_depth_refine_model import DepthSideBranch


# ──────────────────────────────────────────────────────────────────────────────
# LoRA adapter for frozen Linear layers
# ──────────────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Low-rank trainable adapter wrapping a frozen Linear.

        y = W x + b  +  (alpha / r) * (B (A x))

    The original W and b stay frozen (requires_grad=False); only A (r x in)
    and B (out x r) are learned. B is zero-initialised so the adapter starts
    as identity (delta = 0) and the forward pass matches the frozen baseline
    at step 0.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0) -> None:
        super().__init__()
        assert rank > 0, "LoRA rank must be positive"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_features  = base.in_features
        out_features = base.out_features
        self.lora_A  = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B  = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.scale   = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base output preserves frozen behaviour
        out = self.base(x)
        # delta = x @ A^T @ B^T  (broadcast over leading dims)
        delta = torch.matmul(x, self.lora_A.t())
        delta = torch.matmul(delta, self.lora_B.t())
        return out + delta * self.scale


def _apply_lora_to_dino(dino: nn.Module, rank: int, alpha: float = 16.0) -> int:
    """Replace `attn.qkv` Linear in every ViT block with a LoRA-wrapped copy.

    Returns the number of layers wrapped. Raises if no blocks are found
    so silent no-ops don't masquerade as a working LoRA setup.
    """
    blocks = list(getattr(dino, "blocks", []))
    if not blocks:
        raise RuntimeError(
            "LoRA injection failed: DINOv2 encoder has no `.blocks` attribute "
            "(unexpected ViT-L structure)."
        )
    n = 0
    for blk in blocks:
        attn = getattr(blk, "attn", None)
        qkv  = getattr(attn, "qkv", None) if attn is not None else None
        if isinstance(qkv, nn.Linear):
            attn.qkv = LoRALinear(qkv, rank=rank, alpha=alpha)
            n += 1
    if n == 0:
        raise RuntimeError(
            "LoRA injection wrapped 0 layers: `attn.qkv` not found on any block."
        )
    return n


def _unfreeze_last_n_blocks(dino: nn.Module, n: int) -> int:
    """Set requires_grad=True on parameters of the last `n` ViT blocks.

    Returns the number of blocks actually unfrozen. Raises if no blocks are
    found so silent no-ops don't masquerade as a working unfreeze.
    """
    if n <= 0:
        return 0
    blocks = list(getattr(dino, "blocks", []))
    if not blocks:
        raise RuntimeError(
            "Unfreeze failed: DINOv2 encoder has no `.blocks` attribute "
            "(unexpected ViT-L structure)."
        )
    n = min(n, len(blocks))
    for blk in blocks[-n:]:
        for p in blk.parameters():
            p.requires_grad = True
    return n


# ──────────────────────────────────────────────────────────────────────────────
# Plücker ray embeddings
# ──────────────────────────────────────────────────────────────────────────────

class PluckerEmbedder(nn.Module):
    """Per-pixel camera-ray embeddings using Plücker coordinates.

    For each spatial location at resolution (H, W), computes the 3D ray
    through that pixel given camera intrinsics K and world-to-camera T_wc.
    The ray is represented as a 6D Plücker vector [d, o×d] where d is the
    unit ray direction in world space and o is the camera origin.

    Two linear projections map 6D → 256D (for skip connections) and
    6D → 512D (for the bottleneck), both without bias so Plücker embeddings
    start at zero and are learned additively on top of visual features.

    K in the dataset annotations is for the original 1920×1080 capture
    resolution; it is rescaled to the actual (H, W) before unprojection.
    """

    _ORIG_H: int = 1080
    _ORIG_W: int = 1920

    def __init__(self) -> None:
        super().__init__()
        self.proj_256 = nn.Linear(6, 256, bias=False)
        self.proj_512 = nn.Linear(6, 512, bias=False)

    def _rays(
        self,
        H: int,
        W: int,
        K: torch.Tensor,    # (B, 3, 3)  original-resolution intrinsics
        T_wc: torch.Tensor, # (B, 4, 4)  world-to-camera
    ) -> torch.Tensor:      # (B, 6, H, W)
        B, device = K.shape[0], K.device

        # Scale K from original capture resolution to (H, W)
        K_s = K.clone().float()
        K_s[:, 0, :] *= W / self._ORIG_W   # fx, cx
        K_s[:, 1, :] *= H / self._ORIG_H   # fy, cy

        # Pixel grid with half-pixel offset (u=col, v=row)
        us = torch.arange(W, device=device, dtype=torch.float32) + 0.5
        vs = torch.arange(H, device=device, dtype=torch.float32) + 0.5
        gv, gu = torch.meshgrid(vs, us, indexing='ij')          # (H, W)
        pixels = torch.stack(
            [gu, gv, torch.ones(H, W, device=device)], dim=-1
        )                                                         # (H, W, 3)
        pixels = pixels.view(1, H * W, 3).expand(B, -1, -1)     # (B, HW, 3)

        # Unproject to camera-space unit directions
        K_inv = torch.linalg.inv(K_s)                            # (B, 3, 3)
        d_cam = torch.bmm(pixels, K_inv.transpose(1, 2))         # (B, HW, 3)
        d_cam = F.normalize(d_cam, dim=-1)

        # Rotate to world space: R^T @ d_cam  (as row-vectors: d_cam @ R)
        R = T_wc[:, :3, :3].float()                              # (B, 3, 3)
        t = T_wc[:, :3,  3].float()                              # (B, 3)
        d = F.normalize(torch.bmm(d_cam, R), dim=-1)            # (B, HW, 3)

        # Camera origin in world: -R^T @ t  (row-vec: -t @ R)
        o = -torch.bmm(t.unsqueeze(1), R).squeeze(1)             # (B, 3)
        o = o.unsqueeze(1).expand(-1, H * W, -1)                 # (B, HW, 3)

        # Plücker moment: m = o × d
        m = torch.linalg.cross(o, d, dim=-1)                    # (B, HW, 3)

        plucker = torch.cat([d, m], dim=-1)                      # (B, HW, 6)
        return plucker.reshape(B, H, W, 6).permute(0, 3, 1, 2)  # (B, 6, H, W)

    def _project(self, rays: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """(B, 6, H, W) → (B, C, H, W)"""
        return proj(rays.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(
        self,
        H: int,
        W: int,
        K: torch.Tensor,         # (B, 3, 3)
        T_wc: torch.Tensor,      # (B, 4, 4)
        skip_shapes: List[Tuple[int, int]],   # [(Hs, Ws), ...]
        bott_shape:  Tuple[int, int],         # (Hb, Wb)
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns (skip_embs, bott_emb) to add to encoder outputs."""
        rays = self._rays(H, W, K, T_wc)   # (B, 6, H, W)

        skip_embs = []
        for (Hs, Ws) in skip_shapes:
            r = rays if (Hs == H and Ws == W) else \
                F.interpolate(rays, size=(Hs, Ws), mode='bilinear', align_corners=False)
            skip_embs.append(self._project(r, self.proj_256))

        Hb, Wb = bott_shape
        rb = F.interpolate(rays, size=(Hb, Wb), mode='bilinear', align_corners=False)
        bott_emb = self._project(rb, self.proj_512)

        return skip_embs, bott_emb


# ──────────────────────────────────────────────────────────────────────────────
# Depth-only DINO model: DepthSideBranch encoder + DINODecoder
# ──────────────────────────────────────────────────────────────────────────────

class StereoDepthDINOUNet(nn.Module):
    """N-view stereo depth refinement using DepthSideBranch + DINODecoder.

    Inputs  : d_pro  (B, n_views, 1, H, W) — PRO depth only, no RGB
    Output  : (B, n_views, 1, H, W) — refined depth per view

    All parameters are trainable (no frozen backbone).

    Architecture per view
    ─────────────────────
      DepthSideBranch(d_pro_i) → (s0: 32×H, s1: 64×H/2, s2: 128×H/4)
      proj0(s0) → (256×H,   W)   ← skip 0 (full-res)
      proj1(s1) → (256×H/2, W/2) ← skip 1
      proj2(s2) → (256×H/4, W/4) ← skip 2
      maxpool(s2) + proj3 → (256×H/8, W/8)   ← skip 3
      maxpool(s2) + bott_proj → (512×H/8, W/8) ← bottleneck

    Cross-view fusion (when not no_fusion)
    ───────────────────────────────────────
      All views' bottlenecks concatenated → 1×1 conv → shared bottleneck
      then each view decoded with its own skips.
    """

    def __init__(
        self,
        n_views:   int  = 2,
        no_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.n_views   = n_views
        self.no_fusion = no_fusion

        # Shared depth encoder
        self.dsb = DepthSideBranch()

        # Project DSB outputs to 256-ch (required by DINODecoder)
        self.proj0 = nn.Conv2d(32,  256, kernel_size=1)  # s0: (32,H,W)    → (256,H,W)
        self.proj1 = nn.Conv2d(64,  256, kernel_size=1)  # s1: (64,H/2,W/2)→ (256,H/2,W/2)
        self.proj2 = nn.Conv2d(128, 256, kernel_size=1)  # s2: (128,H/4,W/4)→(256,H/4,W/4)
        self.proj3 = nn.Conv2d(128, 256, kernel_size=1)  # maxpool(s2)      → (256,H/8,W/8)
        self.bott_proj = nn.Conv2d(128, 512, kernel_size=1)  # bottleneck  → (512,H/8,W/8)
        self.pool  = nn.MaxPool2d(2)

        # Cross-view bottleneck fusion
        if not no_fusion:
            self.fuse_conv = nn.Conv2d(n_views * 512, 512, kernel_size=1)

        # One shared decoder
        self.decoder = DINODecoder()

    def _encode(self, d: torch.Tensor):
        """(B,1,H,W) → bottleneck (B,512,H/8,W/8), skips list (high→low res)."""
        s0, s1, s2 = self.dsb(d)          # (B,32,H,W), (B,64,H/2,W/2), (B,128,H/4,W/4)
        s2p = self.pool(s2)               # (B,128,H/8,W/8)
        bott  = self.bott_proj(s2p)       # (B,512,H/8,W/8)
        skips = [
            self.proj0(s0),               # (B,256,H,W)    ← full-res, used last
            self.proj1(s1),               # (B,256,H/2,W/2)
            self.proj2(s2),               # (B,256,H/4,W/4)
            self.proj3(s2p),              # (B,256,H/8,W/8) ← lowest-res, used first
        ]
        return bott, skips

    def forward(
        self,
        d_pro:  torch.Tensor,            # (B, n_views, 1, H, W)
        pose_T: torch.Tensor | None = None,  # unused — kept for API compat
    ) -> torch.Tensor:                   # (B, n_views, 1, H, W)
        B, V = d_pro.shape[:2]

        bottlenecks: List[torch.Tensor] = []
        all_skips:   List[List]         = []

        for i in range(V):
            bott, skips = self._encode(d_pro[:, i])
            bottlenecks.append(bott)
            all_skips.append(skips)

        if self.no_fusion:
            preds = [
                F.softplus(self.decoder(bottlenecks[i], all_skips[i]).clamp(-20, 20))
                for i in range(V)
            ]
        else:
            b_fused = self.fuse_conv(torch.cat(bottlenecks, dim=1))
            preds = [
                F.softplus(self.decoder(b_fused, all_skips[i]).clamp(-20, 20))
                for i in range(V)
            ]

        return torch.stack(preds, dim=1)   # (B, n_views, 1, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Model (RGB + depth)
# ──────────────────────────────────────────────────────────────────────────────

class MVStereoDINOUNet(nn.Module):
    """N-view stereo depth refinement with DINOv2 ViT-L encoder.

    Drop-in replacement for MVStereoUNet — same interface, same pose encoding,
    same loss functions.  The only change is the encoder/decoder backbone.

    Inputs
    ------
      rgb    : (B, n_views, 3, H, W)
      d_pro  : (B, n_views, 1, H, W)   PRO depth, median-scaled
      pose_T : (B, n_views, 4, 4)      world-to-camera (OpenCV)

    Output
    ------
      (B, n_views, 1, H, W)  refined depth per view

    Trainable parameters
    ────────────────────
      DINOv2 backbone is frozen.  Trainable: depth side branch, skip fusion
      projections, pose MLP, fuse_conv, decoder (~5 M params).
    """

    def __init__(
        self,
        n_views:    int  = 2,
        use_pose:   bool = True,   # False → no pose embedding at bottleneck
        pose_ch:    int  = 32,
        no_fusion:  bool = False,
        use_plucker: bool = False,  # True → add Plücker ray embeddings to skips + bottleneck
        unfreeze_last_n: int = 0,   # >0 → mark last N DINOv2 blocks trainable
        lora_rank:       int = 0,   # >0 → wrap every block's attn.qkv with a LoRA adapter
        lora_alpha:      float = 16.0,
        pred_mode:       str = "absolute",   # "absolute" | "bounded_residual" | "adaptive_residual"
        max_delta:       float = 1.0,
    ) -> None:
        super().__init__()
        self.n_views    = n_views
        self.no_fusion  = no_fusion
        self.use_pose   = use_pose and not no_fusion
        self.use_plucker = use_plucker

        assert pred_mode in ("absolute", "bounded_residual", "adaptive_residual"), \
            f"pred_mode must be one of absolute/bounded_residual/adaptive_residual, got {pred_mode}"
        self.pred_mode = pred_mode
        self.max_delta = float(max_delta)

        # Frozen DINOv2 ViT-L + trainable depth branch + skip projections
        # NOTE: RGB is always required for DINOv2. For depth-only experiments
        # use MVStereoUNet with use_rgb=False instead.
        self.encoder = DINOv2ViTLEncoder()
        Cb = self.encoder.out_ch   # 512

        # Optional partial fine-tuning of the DINOv2 backbone.
        # The encoder constructor freezes everything; these two paths add back
        # selectively trainable parameters without changing the base weights.
        if unfreeze_last_n > 0:
            n_unfrozen = _unfreeze_last_n_blocks(self.encoder.dino, unfreeze_last_n)
            print(f"[MVStereoDINOUNet] unfroze last {n_unfrozen} DINOv2 blocks")
        if lora_rank > 0:
            n_lora = _apply_lora_to_dino(
                self.encoder.dino, rank=lora_rank, alpha=lora_alpha
            )
            print(f"[MVStereoDINOUNet] wrapped {n_lora} attn.qkv layers with "
                  f"LoRA(r={lora_rank}, alpha={lora_alpha})")

        if not no_fusion:
            if self.use_pose:
                self.pose_project = PoseProject(in_dim=3, pose_ch=pose_ch)
                fuse_in = n_views * (Cb + pose_ch)
            else:
                fuse_in = n_views * Cb
            self.fuse_conv = nn.Conv2d(fuse_in, Cb, kernel_size=1, bias=True)

        if use_plucker:
            self.plucker = PluckerEmbedder()

        # DINODecoder: uniform 256-ch skips, outputs (B, 1, H, W)
        self.decoder = DINODecoder()

        if pred_mode == "adaptive_residual":
            # Replace the decoder's final 1x1 Conv2d(C->1) head with a
            # 2-channel head. Zero-init weights and bias so the model starts
            # at delta=0, gate=sigmoid(0)=0.5 -> adaptive_delta=0 -> pred = d_pro.
            head = self.decoder.head
            assert isinstance(head, nn.Conv2d), \
                f"expected decoder.head to be Conv2d, got {type(head)}"
            new_head = nn.Conv2d(
                head.in_channels, 2,
                kernel_size=head.kernel_size,
                stride=head.stride,
                padding=head.padding,
                bias=(head.bias is not None),
            )
            nn.init.zeros_(new_head.weight)
            if new_head.bias is not None:
                nn.init.zeros_(new_head.bias)
            self.decoder.head = new_head

        self._last_aux: dict | None = None

    def _apply_pred_head(
        self,
        raw:     torch.Tensor,
        d_pro_i: torch.Tensor,
    ) -> tuple[torch.Tensor, dict | None]:
        if self.pred_mode == "absolute":
            return F.softplus(raw.clamp(-20, 20)).clamp(min=1e-3), None
        if self.pred_mode == "bounded_residual":
            raw_clamped = raw.clamp(-20, 20)
            delta = self.max_delta * torch.tanh(raw_clamped)
            pred  = d_pro_i + delta
            return pred.clamp(min=1e-3), {"raw_delta": raw_clamped, "delta": delta}
        # adaptive_residual
        raw_delta = raw[:, 0:1].clamp(-20, 20)
        raw_gate  = raw[:, 1:2].clamp(-20, 20)
        delta     = self.max_delta * torch.tanh(raw_delta)
        gate      = torch.sigmoid(raw_gate)
        adaptive_delta = gate * delta
        pred = d_pro_i + adaptive_delta
        aux  = {
            "raw_delta":      raw_delta,
            "delta":          delta,
            "gate":           gate,
            "adaptive_delta": adaptive_delta,
        }
        return pred.clamp(min=1e-3), aux

    def _pose_encoding(self, pose_T: torch.Tensor) -> torch.Tensor:
        """(B, V, 4, 4) → (B, V, 3) absolute 3-scalar pose vectors."""
        delta_z = pose_T[:, :, 2, 3] - _CAM_Z_BASE

        rot_z       = torch.atan2(pose_T[:, :, 1, 0], pose_T[:, :, 0, 0])
        delta_rot_z = rot_z - _CAM_ROT_Z_BASE
        delta_rot_z = (delta_rot_z + math.pi) % (2 * math.pi) - math.pi

        r_camera = torch.sqrt(
            pose_T[:, :, 0, 3] ** 2 + pose_T[:, :, 1, 3] ** 2
        )
        delta_r = r_camera - _CAM_R_BASE

        return torch.stack([delta_z, delta_rot_z, delta_r], dim=-1)  # (B, V, 3)

    def forward(
        self,
        rgb:    torch.Tensor,                # (B, n_views, 3, H, W)
        d_pro:  torch.Tensor,                # (B, n_views, 1, H, W)
        pose_T: torch.Tensor | None = None,  # (B, n_views, 4, 4); ignored when use_pose=False
        K:      torch.Tensor | None = None,  # (B, n_views, 3, 3); required when use_plucker=True
    ) -> torch.Tensor:                       # (B, n_views, 1, H, W)
        B, V, _, H, W = rgb.shape

        bottlenecks: List[torch.Tensor] = []
        all_skips:   List[List]         = []

        for i in range(V):
            b_i, skips_i = self.encoder(rgb[:, i], d_pro[:, i])

            if self.use_plucker and K is not None and pose_T is not None:
                skip_shapes = [(s.shape[2], s.shape[3]) for s in skips_i]
                bott_shape  = (b_i.shape[2], b_i.shape[3])
                skip_embs, bott_emb = self.plucker(
                    H, W, K[:, i], pose_T[:, i], skip_shapes, bott_shape
                )
                skips_i = [s + e for s, e in zip(skips_i, skip_embs)]
                b_i     = b_i + bott_emb

            bottlenecks.append(b_i)
            all_skips.append(skips_i)

        preds: List[torch.Tensor] = []
        aux_per_view: List[dict | None] = []

        if self.no_fusion:
            for i, (b_i, sk) in enumerate(zip(bottlenecks, all_skips)):
                raw = self.decoder(b_i, sk)
                pred_i, aux_i = self._apply_pred_head(raw, d_pro[:, i])
                preds.append(pred_i); aux_per_view.append(aux_i)
        else:
            if self.use_pose:
                pose_vecs = self._pose_encoding(pose_T)   # (B, V, 3)
                fused_parts: List[torch.Tensor] = []
                for i in range(V):
                    p_i = self.pose_project(pose_vecs[:, i])
                    h, w = bottlenecks[i].shape[2], bottlenecks[i].shape[3]
                    p_i  = p_i.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
                    fused_parts.append(torch.cat([bottlenecks[i], p_i], dim=1))
            else:
                fused_parts = bottlenecks
            b_fused = self.fuse_conv(torch.cat(fused_parts, dim=1))
            for i, sk in enumerate(all_skips):
                raw = self.decoder(b_fused, sk)
                pred_i, aux_i = self._apply_pred_head(raw, d_pro[:, i])
                preds.append(pred_i); aux_per_view.append(aux_i)

        if all(a is not None for a in aux_per_view):
            keys = aux_per_view[0].keys()
            self._last_aux = {
                k: torch.stack([a[k].detach() for a in aux_per_view], dim=1)
                for k in keys
            }
        else:
            self._last_aux = None

        return torch.stack(preds, dim=1)   # (B, n_views, 1, H, W)
