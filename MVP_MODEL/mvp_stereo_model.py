"""
MVP_MODEL/mvp_stereo_model.py
─────────────────────────────
2-view stereo UNet for joint depth refinement of L/R optical-flow image pairs.

Architecture
────────────
  • Shared UNet encoder (in_ch=4: RGB + PRO depth) encodes each view.
  • Per-view bottleneck concatenated with absolute pose embedding.
  • All 2 views fused at bottleneck with a 1×1 conv.
  • Shared UNet decoder produces a refined depth map per view.
  • Output: (B, 2, 1, H, W)

Pose encoding (same as MVUNetPoseConcat)
────────────────────────────────────────
  3 scalars relative to box/shot01 reference camera:
    delta_z     = T_wc[2,3] - 0.85          (height, m)
    delta_rot_z = atan2(R[1,0],R[0,0]) + 3.0885, wrapped to [-π, π]  (azimuth, rad)
    delta_r     = sqrt(T_wc[0,3]^2 + T_wc[1,3]^2) - 12.148          (radial dist, m)
  Projected via 2-layer MLP → pose_ch=32 embedding, broadcast to bottleneck spatial dims.

Losses
──────
  silog_loss_2view              — scale-invariant log loss on masked GT depth (both views).
  stereo_mv_consistency_loss    — bidirectional GT-warp log-difference loss (L→R + R→L).
  masked_rmse_2view             — per-view and average RMSE on masked pixels (for reporting).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse building blocks from the 3-view UNet model
from mvp_depth_model_Unet import (
    ConvBlock, Down, Up, UNetEncoder, UNetDecoder,
    PoseProject,
    _CAM_Z_BASE, _CAM_ROT_Z_BASE, _CAM_R_BASE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers (same logic as train_depth_da2_mv.py)
# ──────────────────────────────────────────────────────────────────────────────

def _make_pixel_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
    """(3, H*W) homogeneous pixel coordinate grid [u, v, 1]^T per column."""
    v, u = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij',
    )
    return torch.stack([u, v, torch.ones_like(u)], dim=0).reshape(3, H * W)


def _camera_baseline(T_i_wc: torch.Tensor, T_j_wc: torch.Tensor) -> torch.Tensor:
    """Per-batch Euclidean distance (m) between two camera centres."""
    C_i = torch.inverse(T_i_wc)[:, :3, 3]
    C_j = torch.inverse(T_j_wc)[:, :3, 3]
    return torch.norm(C_i - C_j, dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Spatial-gated fusion module (per-pair softmax weights, per spatial location)
# ──────────────────────────────────────────────────────────────────────────────

class SpatialGatedPairFusion(nn.Module):
    """Per-location softmax-weighted average over pair features.

    Given a list of pair feature maps each shaped (B, C, H, W), produce a single
    fused (B, C, H, W) map plus the (B, num_pairs, H, W) weight tensor.
    """

    def __init__(self, channels: int, num_pairs: int):
        super().__init__()
        self.num_pairs = num_pairs
        self.gate = nn.Conv2d(channels * num_pairs, num_pairs, kernel_size=1)

    def forward(self, pair_features):
        assert len(pair_features) == self.num_pairs, \
            f"expected {self.num_pairs} pair features, got {len(pair_features)}"
        stacked = torch.cat(pair_features, dim=1)            # (B, C*P, H, W)
        logits  = self.gate(stacked)                         # (B, P, H, W)
        weights = F.softmax(logits, dim=1)                   # (B, P, H, W)
        fused = torch.zeros_like(pair_features[0])
        for i in range(self.num_pairs):
            fused = fused + weights[:, i:i + 1] * pair_features[i]
        return fused, weights


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class MVStereoUNet(nn.Module):
    """N-view stereo UNet with absolute pose embedding at bottleneck.

    n_views controls how many total views are fused:
      n_views=2  → single stereo pair  (L, R)
      n_views=6  → three stereo pairs  (L1,R1, L2,R2, L3,R3)

    Inputs
    ------
      rgb    : (B, n_views, 3, H, W)  — omit / pass None when use_rgb=False
      d_pro  : (B, n_views, 1, H, W)  PRO depth, median-scaled
      pose_T : (B, n_views, 4, 4)     world-to-camera (OpenCV) — omit / pass None
                                       when use_pose=False

    Output
    ------
      (B, n_views, 1, H, W)  refined depth per view

    Configuration flags
    -------------------
      use_rgb  : include RGB in the per-view encoder input (in_ch 4 vs 1)
      use_pose : inject absolute-pose embedding at bottleneck before fusion
      no_fusion: decode each view independently — disables pose + cross-view mixing
    """

    def __init__(
        self,
        n_views:   int  = 2,
        use_rgb:   bool = True,   # False → depth-only encoder (in_ch=1)
        use_pose:  bool = True,   # False → no pose embedding at bottleneck
        base:      int  = 64,
        pose_ch:   int  = 32,
        no_fusion: bool = False,
        pred_mode: str  = "absolute",   # "absolute" | "bounded_residual" | "adaptive_residual"
        max_delta: float = 1.0,
        fusion:    str  = "concat",     # "concat" (default) | "spatial_gated"
    ) -> None:
        super().__init__()
        self.n_views   = n_views
        self.no_fusion = no_fusion
        self.use_rgb   = use_rgb
        self.use_pose  = use_pose and not no_fusion

        assert pred_mode in ("absolute", "bounded_residual", "adaptive_residual"), \
            f"pred_mode must be one of absolute/bounded_residual/adaptive_residual, got {pred_mode}"
        assert fusion in ("concat", "spatial_gated"), \
            f"fusion must be 'concat' or 'spatial_gated', got {fusion}"
        self.fusion    = fusion
        self.pred_mode = pred_mode
        self.max_delta = float(max_delta)

        in_ch          = 4 if use_rgb else 1
        self.encoder   = UNetEncoder(in_ch=in_ch, base=base)
        Cb             = self.encoder.out_ch
        if not no_fusion:
            if self.fusion == "spatial_gated":
                # Per-spatial softmax-weighted average over pair features.
                # Each pair = L + R views averaged at bottleneck (pose is ignored
                # in this mode for simplicity; --no_pose is expected by callers).
                assert n_views % 2 == 0, "spatial_gated requires even n_views (pairs)"
                if self.use_pose:
                    raise ValueError(
                        "fusion='spatial_gated' is incompatible with use_pose=True; "
                        "pass --no_pose."
                    )
                self.n_pairs_runtime = n_views // 2
                self.spatial_gated_fusion = SpatialGatedPairFusion(
                    channels=Cb, num_pairs=self.n_pairs_runtime
                )
            else:
                if self.use_pose:
                    self.pose_project = PoseProject(in_dim=3, pose_ch=pose_ch)
                    fuse_in = n_views * (Cb + pose_ch)
                else:
                    fuse_in = n_views * Cb
                self.fuse_conv = nn.Conv2d(fuse_in, Cb, kernel_size=1, bias=True)
        self.decoder = UNetDecoder(base=base)
        self._init_weights()

        if pred_mode == "adaptive_residual":
            self._install_adaptive_head()

        self._last_aux: dict | None = None

    def _install_adaptive_head(self) -> None:
        """Replace decoder.head (Conv2d C→1) with a 2-channel head.

        Channel 0 → raw_delta, channel 1 → raw_gate. Zero-init the weights
        and bias so at step 0: delta=0, gate=sigmoid(0)=0.5 → adaptive_delta=0
        → pred = d_pro (a no-op refinement). The network then learns when
        and how much to correct.
        """
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

    def _apply_pred_head(
        self,
        raw:     torch.Tensor,    # (B, C, H, W) decoder output
        d_pro_i: torch.Tensor,    # (B, 1, H, W) PRO depth for view i
    ) -> tuple[torch.Tensor, dict | None]:
        if self.pred_mode == "absolute":
            pred = F.softplus(raw.clamp(-20, 20))
            return pred.clamp(min=1e-3), None

        if self.pred_mode == "bounded_residual":
            raw_clamped = raw.clamp(-20, 20)
            delta = self.max_delta * torch.tanh(raw_clamped)
            pred  = d_pro_i + delta
            aux = {"raw_delta": raw_clamped, "delta": delta}
            return pred.clamp(min=1e-3), aux

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

    def _init_weights(self) -> None:
        """Xavier uniform init with gain=0.1 — keeps early activations small."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        rgb:    torch.Tensor,            # (B, n_views, 3, H, W); ignored when use_rgb=False
        d_pro:  torch.Tensor,            # (B, n_views, 1, H, W)
        pose_T: torch.Tensor | None = None,  # (B, n_views, 4, 4); ignored when use_pose=False
    ) -> torch.Tensor:                   # (B, n_views, 1, H, W)
        B, V = d_pro.shape[:2]

        bottlenecks: List[torch.Tensor] = []
        all_skips:   List[Tuple]        = []

        for i in range(V):
            if self.use_rgb:
                x_i = torch.cat([rgb[:, i], d_pro[:, i]], dim=1)  # (B, 4, H, W)
            else:
                x_i = d_pro[:, i]                                  # (B, 1, H, W)
            b_i, skips_i = self.encoder(x_i)
            bottlenecks.append(b_i)
            all_skips.append(skips_i)

        preds: List[torch.Tensor] = []
        aux_per_view: List[dict | None] = []

        pair_weights = None  # set only when fusion == "spatial_gated"

        if self.no_fusion:
            for i in range(V):
                raw = self.decoder(bottlenecks[i], all_skips[i])
                pred_i, aux_i = self._apply_pred_head(raw, d_pro[:, i])
                preds.append(pred_i); aux_per_view.append(aux_i)
        elif self.fusion == "spatial_gated":
            # Pool L+R bottleneck features per pair, then softmax-gate across pairs.
            n_pairs = V // 2
            pair_feats: List[torch.Tensor] = []
            for p in range(n_pairs):
                pair_feats.append(0.5 * (bottlenecks[2 * p] + bottlenecks[2 * p + 1]))
            b_fused, pair_weights = self.spatial_gated_fusion(pair_feats)
            for i in range(V):
                raw = self.decoder(b_fused, all_skips[i])
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
            for i in range(V):
                raw = self.decoder(b_fused, all_skips[i])
                pred_i, aux_i = self._apply_pred_head(raw, d_pro[:, i])
                preds.append(pred_i); aux_per_view.append(aux_i)

        # Stash aux tensors (stacked over views) for the trainer to log.
        # Detach so the trainer can read them under torch.no_grad without
        # tripping the "tensor.requires_grad -> float()" warning.
        aux_out: dict = {}
        if all(a is not None for a in aux_per_view):
            keys = aux_per_view[0].keys()
            aux_out.update({
                k: torch.stack([a[k].detach() for a in aux_per_view], dim=1)
                for k in keys
            })
        if pair_weights is not None:
            aux_out["pair_weights"] = pair_weights.detach()   # (B, n_pairs, H, W)
        self._last_aux = aux_out if aux_out else None

        return torch.stack(preds, dim=1)   # (B, n_views, 1, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────────────────────────────────────

def silog_loss_nview(
    d_pred: torch.Tensor,       # (B, V, 1, H, W)
    d_gt:   torch.Tensor,       # (B, V, 1, H, W)
    mask:   torch.Tensor,       # (B, V, 1, H, W)
    variance_focus: float = 0.85,
    eps: float = 1e-6,
    min_depth: float = 0.5,
    max_depth: float = 10.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Scale-invariant log loss summed over all V views.

    Only trunk-masked pixels whose GT depth falls in [min_depth, max_depth]
    contribute.  This excludes background seen through canopy gaps (Blender
    reports their true far depth inside the silhouette mask).
    """
    V = d_pred.shape[1]
    per_view: List[torch.Tensor] = []
    for v in range(V):
        pred  = d_pred[:, v].clamp(min=eps)
        gt    = d_gt[:, v].clamp(min=eps)
        m     = (mask[:, v].float()
                 * (d_gt[:, v] > min_depth).float()
                 * (d_gt[:, v] < max_depth).float())
        n     = m.sum() + eps
        d     = (torch.log(pred) - torch.log(gt)) * m
        var   = (d ** 2).sum() / n - variance_focus * (d.sum() / n) ** 2
        per_view.append(torch.sqrt(var.clamp(min=0.0)))
    total = sum(per_view)
    return total, per_view


# Keep 2-view alias for backwards compatibility
silog_loss_2view = silog_loss_nview

def stereo_mv_consistency_loss(
    d_pred: torch.Tensor,   # (B, V, 1, H, W)  V = 2*n_pairs (views flat)
    d_gt:   torch.Tensor,   # (B, V, 1, H, W)
    mask:   torch.Tensor,   # (B, V, 1, H, W)
    K:      torch.Tensor,   # (B, V, 3, 3)
    T_wc:   torch.Tensor,   # (B, V, 4, 4)
    min_depth:    float = 0.001,
    max_depth:    float = 20.0,
    max_baseline: float = 0.80,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Bidirectional GT-warp MV consistency loss applied within each stereo pair.

    Views are expected flat: views 0,1 = pair 0 (L,R); 2,3 = pair 1; 4,5 = pair 2 …
    For each pair, two directions are computed:
      L→R: warp GT_L into R frame, compare pred_R vs z_R
      R→L: warp GT_R into L frame, compare pred_L vs z_L

    Gradient flows through d_pred only; d_gt is used purely for geometry.
    """
    device = d_pred.device
    B, V, _, H, W = d_pred.shape
    n_pairs = V // 2

    pix = _make_pixel_grid(H, W, device)  # (3, H*W)
    total        = torch.tensor(0.0, device=device)
    n_valid_dirs = 0

    for p in range(n_pairs):
        i_l, i_r = 2 * p, 2 * p + 1

        pred1 = d_pred[:, i_l, 0];  pred2 = d_pred[:, i_r, 0]
        gt1   = d_gt[:,   i_l, 0];  gt2   = d_gt[:,   i_r, 0]
        m1    = mask[:,   i_l, 0];  m2    = mask[:,   i_r, 0]
        K1, K2       = K[:,    i_l], K[:,    i_r]
        T1_wc, T2_wc = T_wc[:, i_l], T_wc[:, i_r]

        baseline = _camera_baseline(T1_wc, T2_wc)
        close    = baseline <= max_baseline
        if not close.any():
            continue

        # Direction pairs: (src_gt, src_mask, src_K, src_T, tgt_pred, tgt_K, tgt_T, tgt_mask)
        directions = [
            (gt2, m2, K2, T2_wc, pred1, K1, T1_wc, m1),  # R→L
            (gt1, m1, K1, T1_wc, pred2, K2, T2_wc, m2),  # L→R
        ]

        for (gt_src, mask_src, K_src, T_src, pred_tgt, K_tgt, T_tgt, mask_tgt) in directions:
            K_src_inv = torch.inverse(K_src)
            T_src_cw  = torch.inverse(T_src)
            T_st      = T_tgt @ T_src_cw

            gt_flat   = gt_src.reshape(B, 1, H * W)
            pts_src   = (K_src_inv @ pix.unsqueeze(0)) * gt_flat

            ones      = torch.ones(B, 1, H * W, device=device, dtype=pts_src.dtype)
            pts_src_h = torch.cat([pts_src, ones], dim=1)
            pts_tgt_h = T_st @ pts_src_h
            pts_tgt   = pts_tgt_h[:, :3]
            z_tgt     = pts_tgt[:, 2:3]

            proj = K_tgt @ pts_tgt
            u    = proj[:, 0:1] / (z_tgt + eps)
            v    = proj[:, 1:2] / (z_tgt + eps)

            u_n  = 2.0 * u / (W - 1) - 1.0
            v_n  = 2.0 * v / (H - 1) - 1.0
            grid = torch.cat([u_n, v_n], dim=1).permute(0, 2, 1).reshape(B, H, W, 2)

            pred_tgt_sampled = F.grid_sample(
                pred_tgt.unsqueeze(1), grid,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            ).squeeze(1)

            mask_tgt_sampled = F.grid_sample(
                mask_tgt.unsqueeze(1).float(), grid,
                mode='nearest', padding_mode='zeros', align_corners=True,
            ).squeeze(1)

            z_map = z_tgt.reshape(B, H, W)
            in_bounds = (
                (u_n.reshape(B, H, W) >= -1) & (u_n.reshape(B, H, W) <= 1) &
                (v_n.reshape(B, H, W) >= -1) & (v_n.reshape(B, H, W) <= 1)
            )
            valid = (
                in_bounds &
                (mask_src > 0.5) &
                (mask_tgt_sampled > 0.5) &
                (gt_src > min_depth) & (gt_src < max_depth) &
                (z_map > min_depth) & (z_map < max_depth) &
                (pred_tgt_sampled > min_depth) & (pred_tgt_sampled < max_depth)
            ) & close.view(B, 1, 1)

            if valid.sum() < 10:
                continue

            log_diff = torch.abs(
                torch.log(pred_tgt_sampled[valid]) - torch.log(z_map[valid])
            )
            total        = total + log_diff.mean()
            n_valid_dirs += 1

    return total / max(n_valid_dirs, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def masked_rmse_nview(
    d_pred: torch.Tensor,   # (B, V, 1, H, W)
    d_gt:   torch.Tensor,   # (B, V, 1, H, W)
    mask:   torch.Tensor,   # (B, V, 1, H, W)
    eps: float = 1e-6,
    min_depth: float = 0.5,
    max_depth: float = 10.0,
) -> Tuple[float, List[float]]:
    """RMSE on trunk-masked, depth-valid pixels, per view and average."""
    V = d_pred.shape[1]
    per_view: List[float] = []
    for v in range(V):
        m     = (mask[:, v].float()
                 * (d_gt[:, v] > min_depth).float()
                 * (d_gt[:, v] < max_depth).float())
        diff2 = ((d_pred[:, v] - d_gt[:, v]) ** 2) * m
        rmse  = torch.sqrt(diff2.sum() / (m.sum() + eps))
        per_view.append(rmse.item())
    return sum(per_view) / V, per_view


# Keep 2-view alias for backwards compatibility
masked_rmse_2view = masked_rmse_nview
