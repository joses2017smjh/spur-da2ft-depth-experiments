"""
dataset/trunk_stereo_triplet_mvp.py
────────────────────────────────────
Dataset for 6-view (3 stereo pairs) depth refinement using MVStereoUNet(n_views=6).

Pairing strategy (mirrors TrunkDA2MV)
──────────────────────────────────────
  • Group manifest rows by (bark, tree, shot) — same orbital position, different
    camera rigs (e.g. box, box_cam1, box_cam2, box_cam3, box_cam4).
  • At init, precompute per-row nearest neighbours within each scene group by
    Euclidean camera-center distance (loaded from annotation JSONs).
  • At sample time: for each primary row, draw 2 nearest neighbours using
    inverse-distance weighting (closer rigs sampled more often).
  • Each of the 3 rows contributes a stereo L/R pair → 6 views flat:
      L1, R1, L2, R2, L3, R3.

All 3 rigs are at the same (bark, tree, shot), so they have guaranteed
geometric overlap.  The max_baseline guard in stereo_mv_consistency_loss
provides an additional runtime check.

Returned batch keys
───────────────────
  rgb    : (6, 3, H, W)    ImageNet-normalised RGB
  d_pro  : (6, 1, H, W)    PRO depth (median-scaled to GT on trunk pixels)
  d_gt   : (6, 1, H, W)    GT depth
  mask   : (6, 1, H, W)    trunk mask
  K      : (6, 3, 3)       camera intrinsics
  T_wc   : (6, 4, 4)       world-to-camera (OpenCV)
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize, to_tensor


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_DEPTH_INF_THRESH = 1e9

_DEPTH_SUBDIR_GT  = os.sep + "depth"      + os.sep
# Input depth subdir is overridable via INPUT_DEPTH_SUBDIR env (e.g. Da2Finetune).
# Mirrors trunk_stereo_mvp.py so triplet runs can swap input-depth source the same way.
_DEPTH_SUBDIR_PRO = os.sep + os.environ.get("INPUT_DEPTH_SUBDIR", "pro_refine") + os.sep
_INPUT_DEPTH_INTERP = os.environ.get("INPUT_DEPTH_INTERP", "bilinear")  # "nearest" avoids trunk/bg edge-averaging (bad pixels)

# PRO depth calibration — from Baseline_Model/eval/benchmark_summary.json
# pro.best_config: global fit, erode_r=10, min_gt_std=0.05, fit_space=depth
_PRO_ALPHA     = -0.06610793956568871
_PRO_BETA      =  1.555980697834118
_PRO_DEPTH_EPS =  1e-3


# ──────────────────────────────────────────────────────────────────────────────
# Pose loading
# ──────────────────────────────────────────────────────────────────────────────

def _euler_xyz_to_T(location: list, rotation_euler: list) -> np.ndarray:
    """Blender (location, XYZ euler) → 4×4 world-to-camera in OpenCV convention."""
    rx, ry, rz = rotation_euler
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    T = np.eye(4, dtype=np.float32)
    T[0, 0] = cy * cz;  T[0, 1] = sx * sy * cz - cx * sz;  T[0, 2] = cx * sy * cz + sx * sz
    T[1, 0] = cy * sz;  T[1, 1] = sx * sy * sz + cx * cz;  T[1, 2] = cx * sy * sz - sx * cz
    T[2, 0] = -sy;      T[2, 1] = sx * cy;                  T[2, 2] = cx * cy
    T[0, 3] = location[0]
    T[1, 3] = location[1]
    T[2, 3] = location[2]

    blender_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    T_cw = T @ blender_to_cv
    return np.linalg.inv(T_cw).astype(np.float32)


def _load_pose_from_ann(ann_path: str):
    """Load (K 3×3, T_wc 4×4) from annotation JSON.  Returns (None, None) on failure."""
    try:
        with open(ann_path) as f:
            ann = json.load(f)
        cam = ann["camera"]
        K    = np.array(cam["intrinsics"]["K"], dtype=np.float32)
        T_wc = _euler_xyz_to_T(cam["location"], cam["rotation_euler"])
        return K, T_wc
    except Exception:
        return None, None


def _camera_center(ann_path: str) -> np.ndarray | None:
    """Return world-space camera center (3,) from annotation JSON, or None."""
    try:
        with open(ann_path) as f:
            ann = json.load(f)
        return np.array(ann["camera"]["location"], dtype=np.float64)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TrunkStereoTripletMVPDataset(Dataset):
    """3-pair (6-view) stereo dataset for MVStereoUNet(n_views=6).

    Groups manifest rows by (bark, tree, shot).  Within each group (same
    orbital position, different camera rigs), nearest neighbours are computed
    by Euclidean camera-center distance.  At sample time a primary rig is
    chosen, then 2 nearest rigs are drawn with inverse-distance weighting —
    matching the TrunkDA2MV pairing strategy.

    Requires ≥3 distinct set_ids per (bark, tree, shot) group.

    Args
    ----
    manifest_path  : stereo CSV manifest with pair_* columns
    H, W           : output spatial size
    max_neighbours : how many nearest rigs to sample from (default 5)
    path_remap     : optional "OLD_PREFIX:NEW_PREFIX" to fix manifest paths
    """

    def __init__(
        self,
        manifest_path: str | Path,
        H: int = 280,
        W: int = 512,
        max_neighbours: int = 5,
        path_remap: str | None = None,
        pro_calib: bool = True,
    ) -> None:
        self.H = H
        self.W = W
        self._max_neighbours = max_neighbours
        self.pro_calib = pro_calib

        self._remap_from: str | None = None
        self._remap_to:   str | None = None
        if path_remap:
            parts = path_remap.split(":", 1)
            if len(parts) == 2:
                self._remap_from, self._remap_to = parts

        # ── Load and remap manifest ───────────────────────────────────────────
        with open(manifest_path, newline="") as fh:
            all_rows = list(csv.DictReader(fh))

        if self._remap_from:
            for row in all_rows:
                for col in row:
                    row[col] = row[col].replace(self._remap_from, self._remap_to)

        # ── Filter rows: both views need ann + PRO depth ──────────────────────
        valid_rows: list[dict] = []
        skipped_ann = skipped_pro = 0
        for row in all_rows:
            if not row.get("ann_path") or not row.get("pair_ann_path"):
                skipped_ann += 1
                continue
            pro1 = row["depth_path"].replace(_DEPTH_SUBDIR_GT, _DEPTH_SUBDIR_PRO)
            pro2 = row["pair_depth_path"].replace(_DEPTH_SUBDIR_GT, _DEPTH_SUBDIR_PRO)
            if not os.path.isfile(pro1) or not os.path.isfile(pro2):
                skipped_pro += 1
                continue
            valid_rows.append(row)

        self.rows = valid_rows

        # ── Group by (bark, tree, shot) ───────────────────────────────────────
        scene_groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.rows):
            key = f"{r.get('bark','')}_{r['tree']}_{r['shot']}"
            scene_groups[key].append(i)

        self._row_scene_key: list[str] = [
            f"{r.get('bark','')}_{r['tree']}_{r['shot']}" for r in self.rows
        ]
        self._scene_groups = dict(scene_groups)

        # ── Precompute nearest neighbours by camera-center distance ───────────
        cam_locs: dict[int, np.ndarray] = {}
        for i, r in enumerate(self.rows):
            loc = _camera_center(r["ann_path"])
            if loc is not None:
                cam_locs[i] = loc

        self._nearby: dict[int, list[tuple[int, float]]] = {}
        n_no_nb = 0
        for key, group in self._scene_groups.items():
            with_loc = [g for g in group if g in cam_locs]
            for idx in group:
                if idx not in cam_locs or len(with_loc) < 3:
                    n_no_nb += 1
                    continue
                dists = [
                    (other, float(np.linalg.norm(cam_locs[idx] - cam_locs[other])))
                    for other in with_loc if other != idx
                ]
                dists.sort(key=lambda x: x[1])
                self._nearby[idx] = dists

        # Active index: only rows with ≥2 neighbours
        self._active: list[int] = [i for i in range(len(self.rows)) if i in self._nearby]

        print(f"  {Path(manifest_path).name}: {len(valid_rows):,} valid rows, "
              f"{len(self._active):,} active (≥2 neighbours) "
              f"(skipped {skipped_ann} missing ann, {skipped_pro} missing PRO, "
              f"{n_no_nb} no neighbours)")

    def __len__(self) -> int:
        return len(self._active)

    def __getitem__(self, idx: int) -> dict:
        primary_idx = self._active[idx]

        # ── Draw 2 nearest neighbours with inverse-distance weighting ─────────
        neighbours = self._nearby[primary_idx]
        top_k  = neighbours[: self._max_neighbours]
        dists  = np.array([d for _, d in top_k], dtype=np.float64)
        weights = 1.0 / (dists + 1e-6)
        weights /= weights.sum()

        n_avail = len(top_k)
        if n_avail >= 2:
            chosen = np.random.choice(n_avail, size=2, replace=False, p=weights)
        else:
            chosen = np.array([0, 0])

        row_indices = [primary_idx, top_k[chosen[0]][0], top_k[chosen[1]][0]]

        # ── Load 3 stereo pairs → 6 views flat (L1,R1, L2,R2, L3,R3) ─────────
        all_rgbs, all_d_pros, all_d_gts, all_masks, all_Ks, all_T_wcs = \
            [], [], [], [], [], []

        for ri in row_indices:
            row = self.rows[ri]
            for side in ("primary", "secondary"):
                if side == "primary":
                    vd = {
                        "rgb_path":   row["rgb_path"],
                        "depth_path": row["depth_path"],
                        "mask_path":  row.get("mask_path", ""),
                        "ann_path":   row["ann_path"],
                    }
                else:
                    vd = {
                        "rgb_path":   row["pair_rgb_path"],
                        "depth_path": row["pair_depth_path"],
                        "mask_path":  row.get("pair_mask_path", ""),
                        "ann_path":   row["pair_ann_path"],
                    }

                rgb_t  = self._load_rgb(vd["rgb_path"])
                d_gt_t = self._load_depth(vd["depth_path"])
                mask_t = self._load_mask(vd["mask_path"])

                pro_path = vd["depth_path"].replace(_DEPTH_SUBDIR_GT, _DEPTH_SUBDIR_PRO)
                d_pro_t  = self._load_pro_depth(pro_path, d_gt_t, mask_t)

                K, T_wc = _load_pose_from_ann(vd["ann_path"])
                if K is None:
                    K    = np.eye(3, dtype=np.float32)
                    T_wc = np.eye(4, dtype=np.float32)

                all_rgbs.append(rgb_t)
                all_d_pros.append(d_pro_t)
                all_d_gts.append(d_gt_t)
                all_masks.append(mask_t)
                all_Ks.append(torch.from_numpy(K))
                all_T_wcs.append(torch.from_numpy(T_wc))

        return {
            "rgb":   torch.stack(all_rgbs),     # (6, 3, H, W)
            "d_pro": torch.stack(all_d_pros),   # (6, 1, H, W)
            "d_gt":  torch.stack(all_d_gts),    # (6, 1, H, W)
            "mask":  torch.stack(all_masks),    # (6, 1, H, W)
            "K":     torch.stack(all_Ks),       # (6, 3, 3)
            "T_wc":  torch.stack(all_T_wcs),    # (6, 4, 4)
        }

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_rgb(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.W, self.H), Image.BILINEAR)
        return normalize(to_tensor(img), _IMAGENET_MEAN, _IMAGENET_STD)  # (3, H, W)

    def _load_depth(self, path: str) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        arr[arr >= _DEPTH_INF_THRESH] = 0.0
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        # NEAREST (not bilinear): bilinear averages trunk (~2 m) with zeroed
        # background at the silhouette, corrupting ~5% of trunk-mask edge
        # pixels and inflating masked RMSE ~20x. Same fix as in trunk_stereo_mvp.
        t = F.interpolate(t, size=(self.H, self.W), mode="nearest")
        return t.squeeze(0)   # (1, H, W)

    def _load_mask(self, path: str) -> torch.Tensor:
        if not path or not os.path.isfile(path):
            return torch.ones(1, self.H, self.W, dtype=torch.float32)
        arr = np.array(Image.open(path), dtype=np.float32)
        t   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t   = F.interpolate(t, size=(self.H, self.W), mode="nearest")
        return (t.squeeze(0) > 0).float()   # (1, H, W)

    def _load_pro_depth(
        self,
        path: str,
        d_gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        t   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        if _INPUT_DEPTH_INTERP == "nearest":
            t = F.interpolate(t, size=(self.H, self.W), mode="nearest")
        else:
            t = F.interpolate(t, size=(self.H, self.W), mode="bilinear", align_corners=False)
        t   = t.squeeze(0)   # (1, H, W)

        if self.pro_calib:
            # Apply global calibration: depth_cal = alpha * depth_raw + beta
            # alpha is negative — raw PRO is inversely correlated with GT depth.
            t = _PRO_ALPHA * t + _PRO_BETA
        return t.clamp(min=_PRO_DEPTH_EPS)   # (1, H, W)
