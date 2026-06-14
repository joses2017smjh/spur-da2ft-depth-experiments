"""
dataset/trunk_manifest.py
─────────────────────────
Manifest-based dataset for trunk metric depth fine-tuning.

Each row of the manifest CSV points to one (rgb, depth, mask) triplet.
Curriculum is applied via set_curriculum_epoch(epoch):

    epoch 0–1  →  base images only  (set_id == "base")
    epoch >= 2 →  30 % base / 70 % cam  (randomly resampled to dataset length)

All resizing uses OpenCV:
    RGB    →  INTER_AREA
    Depth  →  INTER_AREA  (full-res .npy → target size)
    Mask   →  INTER_NEAREST
"""

import csv

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TrunkManifest(Dataset):
    """
    Args:
        manifest_path : path to train_manifest.csv or val_manifest.csv
        img_size      : square target resolution (must be divisible by 14)
        mode          : 'train' or 'val'  (informational only)
        min_depth     : minimum valid depth in metres
        max_depth     : maximum valid depth in metres

    Item dict:
        image      : FloatTensor (3, H, W)  ImageNet-normalised
        depth      : FloatTensor (H, W)     metric GT depth in metres
        valid_mask : BoolTensor  (H, W)     trunk pixel AND in [min_depth, max_depth]
    """

    def __init__(
        self,
        manifest_path: str,
        img_size:  int   = 518,
        mode:      str   = "train",
        min_depth: float = 0.001,
        max_depth: float = 20.0,
    ) -> None:
        self.img_size  = img_size
        self.min_depth = min_depth
        self.max_depth = max_depth

        self.rows: list[dict] = []
        with open(manifest_path, newline="") as fh:
            for row in csv.DictReader(fh):
                self.rows.append(row)

        # Precompute index sets by set_id for curriculum
        self._base_idxs = np.array(
            [i for i, r in enumerate(self.rows) if r["set_id"] == "base"],
            dtype=np.int64,
        )
        self._cam_idxs = np.array(
            [i for i, r in enumerate(self.rows) if r["set_id"] != "base"],
            dtype=np.int64,
        )

        # Default: all rows visible (correct for val; train overrides via set_curriculum_epoch)
        self._active = np.arange(len(self.rows), dtype=np.int64)

    # ──────────────────────────────────────────────────────────────────────────
    # Curriculum
    # ──────────────────────────────────────────────────────────────────────────

    def set_curriculum_epoch(self, epoch: int) -> None:
        """
        Update which rows are visible for this epoch.
        Call once per epoch BEFORE constructing the DataLoader.

        epoch 0–1  →  base-only (no cam)
        epoch >= 2 →  30 % base / 70 % cam, total length = full dataset size
                       (sampling with replacement so both pools are always covered)
        """
        n_total = len(self.rows)

        if epoch < 2:
            idx = self._base_idxs.copy()
        else:
            n_base = max(1, round(n_total * 0.30))
            n_cam  = n_total - n_base
            base_s = np.random.choice(self._base_idxs, size=n_base, replace=True)
            cam_s  = np.random.choice(self._cam_idxs,  size=n_cam,  replace=True)
            idx    = np.concatenate([base_s, cam_s])

        np.random.shuffle(idx)
        self._active = idx

    # ──────────────────────────────────────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._active)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[self._active[idx]]
        s   = self.img_size

        # ── RGB: BGR → RGB, INTER_AREA downscale, ImageNet normalise ─────────
        bgr = cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA)
        rgb = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(rgb.transpose(2, 0, 1))  # (3, H, W)

        # ── Depth: full-res .npy → INTER_AREA downscale ──────────────────────
        depth_np = np.load(row["depth_path"]).astype(np.float32)      # (H_src, W_src)
        depth_np = cv2.resize(depth_np, (s, s), interpolation=cv2.INTER_AREA)
        depth    = torch.from_numpy(depth_np)                          # (H, W)

        # ── Mask: INTER_NEAREST (binary, no interpolation artifacts) ─────────
        mask_raw = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.resize(mask_raw, (s, s), interpolation=cv2.INTER_NEAREST)
        trunk    = mask_raw > 0                                        # (H, W) bool

        # valid_mask = trunk pixels within the configured depth range
        valid_mask = torch.from_numpy(
            trunk
            & (depth_np >= self.min_depth)
            & (depth_np <= self.max_depth)
        )  # (H, W) bool

        return {
            "image":      image,       # FloatTensor (3, H, W)
            "depth":      depth,       # FloatTensor (H, W)
            "valid_mask": valid_mask,  # BoolTensor  (H, W)
        }
