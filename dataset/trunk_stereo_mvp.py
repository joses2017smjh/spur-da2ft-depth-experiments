"""
dataset/trunk_stereo_mvp.py
───────────────────────────
Dataset for 2-view stereo depth refinement using the MVP UNet model.

Each row in the stereo manifest represents an L/R optical-flow pair.
This dataset loads:
  • RGB image          (ImageNet-normalised)
  • PRO depth map      (median-scaled to GT on trunk pixels)
  • GT depth map
  • Trunk mask
  • Camera intrinsics K (3×3)
  • World-to-camera pose T_wc (4×4, OpenCV convention)
for BOTH the primary (L) and secondary (R) views.

PRO depth path derivation
─────────────────────────
  GT   depth: .../ depth    /bark/tree/set_id/tree_shotNN_{l,r}.npy
  PRO  depth: .../pro_refine/bark/tree/set_id/tree_shotNN_{l,r}.npy

Optional random swap
────────────────────
  With random_swap=True the L and R sides are randomly exchanged at sample
  time, so the model sees both L-primary and R-primary during training.

Returned batch keys
───────────────────
  rgb    : (2, 3, H, W)    normalised RGB
  d_pro  : (2, 1, H, W)    PRO depth (median-scaled)
  d_gt   : (2, 1, H, W)    GT depth
  mask   : (2, 1, H, W)    trunk mask
  K      : (2, 3, 3)       camera intrinsics
  T_wc   : (2, 4, 4)       world-to-camera (OpenCV)
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
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
# Input ("PRO") depth subdir is overridable via env so the same loader can read
# alternative input-depth sources (e.g. DA2-finetuned predictions under
# Da2Finetune/) without touching the manifests. Defaults to pro_refine so all
# existing experiments are unchanged.
_DEPTH_SUBDIR_PRO = os.sep + os.environ.get("INPUT_DEPTH_SUBDIR", "pro_refine") + os.sep
_INPUT_DEPTH_INTERP = os.environ.get("INPUT_DEPTH_INTERP", "bilinear")  # "nearest" avoids trunk/bg edge-averaging (bad pixels)

# PRO depth calibration — from Baseline_Model/eval/benchmark_summary.json
# pro.best_config: global fit, erode_r=10, min_gt_std=0.05, fit_space=depth
# Raw PRO is inversely scaled vs GT (alpha < 0); calibration corrects scale + sign.
_PRO_ALPHA     = -0.06610793956568871
_PRO_BETA      =  1.555980697834118
_PRO_DEPTH_EPS =  1e-3   # clamp floor after calibration (1 mm)


# ──────────────────────────────────────────────────────────────────────────────
# Pose loading
# ──────────────────────────────────────────────────────────────────────────────

def _euler_xyz_to_T(location: list, rotation_euler: list) -> np.ndarray:
    """Blender (location, XYZ euler) → 4×4 camera-to-world in OpenCV convention."""
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

    # Blender camera → OpenCV: negate Y and Z axes of camera frame
    blender_to_cv = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    T_cw = T @ blender_to_cv

    # World-to-camera
    return np.linalg.inv(T_cw).astype(np.float32)


def _load_pose_from_ann(ann_path: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Load (K 3×3, T_wc 4×4) from annotation JSON."""
    try:
        with open(ann_path) as f:
            ann = json.load(f)
        cam = ann["camera"]
        K_list = cam["intrinsics"]["K"]
        K = np.array(K_list, dtype=np.float32)
        T_wc = _euler_xyz_to_T(cam["location"], cam["rotation_euler"])
        return K, T_wc
    except Exception:
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TrunkStereoMVPDataset(Dataset):
    """Stereo-pair dataset for MVP UNet depth refinement.

    Args
    ----
    manifest_path : path to a stereo CSV manifest.
                    For n_pairs=1, expects legacy columns
                      rgb_path, depth_path, mask_path, ann_path,
                      pair_rgb_path, pair_depth_path, pair_mask_path, pair_ann_path.
                    For n_pairs>1, expects indexed columns per pair i in 1..n_pairs:
                      rgb_path_{i}, depth_path_{i}, mask_path_{i}, ann_path_{i},
                      pair_rgb_path_{i}, pair_depth_path_{i}, pair_mask_path_{i}, pair_ann_path_{i}.
    H, W          : output spatial size
    random_swap   : if True, randomly swap L/R within each pair at sample time (training only)
    path_remap    : optional "OLD_PREFIX:NEW_PREFIX" to fix manifest paths
    pro_calib     : apply global alpha,beta PRO depth calibration in the loader
    n_pairs       : number of stereo pairs per row (output has 2*n_pairs views)
    """

    def __init__(
        self,
        manifest_path: str | Path,
        H: int = 280,
        W: int = 512,
        random_swap: bool = False,
        path_remap: str | None = None,
        pro_calib: bool = True,
        n_pairs: int = 1,
    ) -> None:
        self.H = H
        self.W = W
        self.random_swap = random_swap
        self.pro_calib = pro_calib
        self.n_pairs = int(n_pairs)
        assert self.n_pairs >= 1, "n_pairs must be >= 1"

        self._remap_from: str | None = None
        self._remap_to:   str | None = None
        if path_remap:
            parts = path_remap.split(":", 1)
            if len(parts) == 2:
                self._remap_from, self._remap_to = parts

        # Load and filter manifest
        with open(manifest_path, newline="") as fh:
            all_rows = list(csv.DictReader(fh))

        if self._remap_from:
            for row in all_rows:
                for col in row:
                    row[col] = row[col].replace(self._remap_from, self._remap_to)

        # Per-pair column key suffixes: "" for legacy 1-pair, "_1".."_n" for multi.
        if self.n_pairs == 1:
            self._pair_suffixes = [""]
        else:
            self._pair_suffixes = [f"_{i}" for i in range(1, self.n_pairs + 1)]

        # Keep only rows where every pair has ann files and PRO depth on disk
        self.rows: list[dict] = []
        skipped_ann = 0
        skipped_pro = 0
        for row in all_rows:
            ann_ok = all(
                row.get(f"ann_path{s}") and row.get(f"pair_ann_path{s}")
                for s in self._pair_suffixes
            )
            if not ann_ok:
                skipped_ann += 1
                continue
            pro_ok = True
            for s in self._pair_suffixes:
                pro_p = row[f"depth_path{s}"].replace(_DEPTH_SUBDIR_GT, _DEPTH_SUBDIR_PRO)
                pro_s = row[f"pair_depth_path{s}"].replace(_DEPTH_SUBDIR_GT, _DEPTH_SUBDIR_PRO)
                if not os.path.isfile(pro_p) or not os.path.isfile(pro_s):
                    pro_ok = False
                    break
            if not pro_ok:
                skipped_pro += 1
                continue
            self.rows.append(row)

        print(f"  {Path(manifest_path).name}: {len(self.rows):,} rows "
              f"(skipped {skipped_ann} missing ann, {skipped_pro} missing PRO; "
              f"n_pairs={self.n_pairs}, n_views={2 * self.n_pairs})")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        # Build view descriptors for every stereo pair in this row.
        views: list[dict] = []
        for s in self._pair_suffixes:
            primary = {
                "rgb_path":   row[f"rgb_path{s}"],
                "depth_path": row[f"depth_path{s}"],
                "mask_path":  row.get(f"mask_path{s}", ""),
                "ann_path":   row[f"ann_path{s}"],
            }
            secondary = {
                "rgb_path":   row[f"pair_rgb_path{s}"],
                "depth_path": row[f"pair_depth_path{s}"],
                "mask_path":  row.get(f"pair_mask_path{s}", ""),
                "ann_path":   row[f"pair_ann_path{s}"],
            }
            if self.random_swap and random.random() < 0.5:
                primary, secondary = secondary, primary
            views.append(primary)
            views.append(secondary)

        rgbs, d_pros, d_gts, masks, Ks, T_wcs = [], [], [], [], [], []

        for v in views:
            rgb_t  = self._load_rgb(v["rgb_path"])
            d_gt_t = self._load_depth(v["depth_path"])
            mask_t = self._load_mask(v["mask_path"])

            pro_path = v["depth_path"].replace(_DEPTH_SUBDIR_GT, _DEPTH_SUBDIR_PRO)
            d_pro_t  = self._load_pro_depth(pro_path, d_gt_t, mask_t)

            K, T_wc = _load_pose_from_ann(v["ann_path"])
            if K is None:
                K   = np.eye(3, dtype=np.float32)
                T_wc = np.eye(4, dtype=np.float32)

            rgbs.append(rgb_t)
            d_pros.append(d_pro_t)
            d_gts.append(d_gt_t)
            masks.append(mask_t)
            Ks.append(torch.from_numpy(K))
            T_wcs.append(torch.from_numpy(T_wc))

        return {
            "rgb":   torch.stack(rgbs),     # (V, 3, H, W) where V = 2*n_pairs
            "d_pro": torch.stack(d_pros),   # (V, 1, H, W)
            "d_gt":  torch.stack(d_gts),    # (V, 1, H, W)
            "mask":  torch.stack(masks),    # (V, 1, H, W)
            "K":     torch.stack(Ks),       # (V, 3, 3)
            "T_wc":  torch.stack(T_wcs),    # (V, 4, 4)
        }

    # ── Loaders ───────────────────────────────────────────────────────────────

    def _load_rgb(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.W, self.H), Image.BILINEAR)
        return normalize(to_tensor(img), _IMAGENET_MEAN, _IMAGENET_STD)  # (3, H, W)

    def _load_depth(self, path: str) -> torch.Tensor:
        arr = np.load(path).astype(np.float32)
        arr[arr >= _DEPTH_INF_THRESH] = 0.0
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1, 1, H_orig, W_orig)
        # NEAREST (not bilinear): GT depth has a sharp trunk/background
        # discontinuity (background sentinel zeroed to 0 just above). Bilinear
        # downsampling averages 2 m trunk with 0 m background at the silhouette,
        # corrupting ~5% of trunk-mask edge pixels and inflating masked RMSE
        # ~20x (1.09 m vs 0.056 m on val). Nearest keeps trunk-edge GT exact.
        t = F.interpolate(t, size=(self.H, self.W), mode="nearest")
        return t.squeeze(0)   # (1, H, W)

    def _load_mask(self, path: str) -> torch.Tensor:
        if not path or not os.path.isfile(path):
            # No mask — treat whole image as valid
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
        try:
            arr = np.load(path).astype(np.float32)
        except (ValueError, OSError, EOFError) as e:
            # Truncated / corrupted .npy — fall back to the calibrated constant
            # (PRO_BETA ≈ 1.556 m, the median PRO output for bark_brown_02).
            print(f"[WARN] corrupt PRO depth, using constant fallback: {path} ({e})",
                  flush=True)
            return torch.full((1, self.H, self.W), _PRO_BETA, dtype=torch.float32)

        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        if _INPUT_DEPTH_INTERP == "nearest":
            t = F.interpolate(t, size=(self.H, self.W), mode="nearest")
        else:
            t = F.interpolate(t, size=(self.H, self.W), mode="bilinear", align_corners=False)
        t = t.squeeze(0)   # (1, H, W)

        if self.pro_calib:
            # Apply global calibration: depth_cal = alpha * depth_raw + beta
            # alpha is negative — raw PRO is inversely correlated with GT depth.
            # Pixels where depth_raw > beta/|alpha| ~23.5 will produce depth_cal <= 0;
            # clamp eliminates those and guards the log in SiLog.
            t = _PRO_ALPHA * t + _PRO_BETA
        return t.clamp(min=_PRO_DEPTH_EPS)   # (1, H, W)
