"""
dataset/stereo_depth_pair.py
────────────────────────────
Minimal dataset for stereo depth-pair refinement.

Each sample is a left/right stereo pair of raw depth maps + trunk masks.
No RGB images, no camera poses — depth only.

Manifest CSV must have columns:
    depth_path, pair_depth_path, mask_path, pair_mask_path

__getitem__ returns:
    depth_l  : (H, W) float32 tensor — left raw depth in metres
    depth_r  : (H, W) float32 tensor — right raw depth in metres
    mask_l   : (H, W) float32 tensor — left trunk mask (0 or 1)
    mask_r   : (H, W) float32 tensor — right trunk mask (0 or 1)
"""

import csv
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class StereoPairDepth(Dataset):
    """
    Args:
        manifest_path : path to a CSV with columns
                        depth_path, pair_depth_path, mask_path, pair_mask_path
        size          : (H, W) to resize all maps to (default 512×512)
        mode          : 'train' enables random horizontal flip; 'val' does not
        random_swap   : if True and mode='train', randomly swap L/R at sample
                        time so the model sees both orderings
    """

    def __init__(
        self,
        manifest_path: str,
        size: tuple = (512, 512),
        mode: str = "train",
        random_swap: bool = True,
    ) -> None:
        assert mode in ("train", "val")
        self.H, self.W = size
        self.mode = mode
        self._random_swap = random_swap and (mode == "train")

        self.rows: list[dict] = []
        with open(manifest_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if (row.get("depth_path") and row.get("pair_depth_path") and
                        row.get("mask_path") and row.get("pair_mask_path")):
                    self.rows.append(row)

        print(f"  StereoPairDepth [{mode}]: {len(self.rows):,} pairs from {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_depth(self, path: str) -> np.ndarray:
        d = np.load(path).astype(np.float32)
        if d.shape != (self.H, self.W):
            d = cv2.resize(d, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return d

    def _load_mask(self, path: str) -> np.ndarray:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.ones((self.H, self.W), dtype=np.float32)
        if m.shape != (self.H, self.W):
            m = cv2.resize(m, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return (m > 0).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]

        dl = self._load_depth(row["depth_path"])
        dr = self._load_depth(row["pair_depth_path"])
        ml = self._load_mask(row["mask_path"])
        mr = self._load_mask(row["pair_mask_path"])

        # Random L/R swap: model sees both orderings
        if self._random_swap and random.random() < 0.5:
            dl, dr = dr, dl
            ml, mr = mr, ml

        # Random horizontal flip for training augmentation
        if self.mode == "train" and random.random() < 0.5:
            dl = dl[:, ::-1].copy()
            dr = dr[:, ::-1].copy()
            ml = ml[:, ::-1].copy()
            mr = mr[:, ::-1].copy()

        return {
            "depth_l": torch.from_numpy(dl),
            "depth_r": torch.from_numpy(dr),
            "mask_l":  torch.from_numpy(ml),
            "mask_r":  torch.from_numpy(mr),
        }
