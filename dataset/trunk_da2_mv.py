"""
dataset/trunk_da2_mv.py
───────────────────────
Multi-view extension of TrunkDA2 for multi-view consistency training.

Each __getitem__ returns a PRIMARY sample (identical to TrunkDA2) plus a
SECONDARY sample from a different camera viewing the same (tree, shot).

The secondary view's camera pose is provided as a 4x4 world-to-camera matrix
(T_wc) along with intrinsics, enabling differentiable reprojection in the
training loop.

Returned dict keys:
    image, depth, valid_mask          — primary view (same as TrunkDA2)
    image2, depth2, valid_mask2       — secondary view
    K1, K2                            — (3,3) intrinsics matrices
    T1_wc, T2_wc                     — (4,4) world-to-camera transforms
"""

import csv
import json
import math
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.trunk_da2 import Resize, NormalizeImage, PrepareForNet, Crop


# ──────────────────────────────────────────────────────────────────────────────
# Camera pose utilities
# ──────────────────────────────────────────────────────────────────────────────

def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Blender XYZ Euler angles → 3x3 rotation matrix."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)

    return Rz @ Ry @ Rx


def blender_to_opencv_T_wc(location: list, rotation_euler: list) -> np.ndarray:
    """
    Convert Blender camera (location, rotation_euler) to a 4x4 world-to-camera
    matrix in OpenCV convention (X-right, Y-down, Z-forward).

    Blender convention: X-right, Y-up, Z-backward (for camera).
    OpenCV convention:  X-right, Y-down, Z-forward.
    """
    R_blender = euler_to_rotation_matrix(*rotation_euler)
    t = np.array(location, dtype=np.float64)

    # Camera-to-world in Blender coordinates
    T_cw_blender = np.eye(4, dtype=np.float64)
    T_cw_blender[:3, :3] = R_blender
    T_cw_blender[:3, 3] = t

    # Blender-to-OpenCV flip: negate Y and Z axes of the camera frame
    flip = np.diag([1.0, -1.0, -1.0, 1.0])

    # Camera-to-world in OpenCV convention
    T_cw_cv = T_cw_blender @ flip

    # World-to-camera
    T_wc = np.linalg.inv(T_cw_cv)
    return T_wc.astype(np.float32)


def load_camera_from_ann(ann_path: str):
    """
    Load camera intrinsics K (3x3) and world-to-camera T_wc (4x4)
    from an annotation JSON file.

    Returns (K, T_wc) as float32 numpy arrays, or (None, None) if missing.
    """
    try:
        with open(ann_path) as f:
            ann = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None

    cam = ann.get("camera", {})
    K_list = cam.get("intrinsics", {}).get("K")
    loc = cam.get("location")
    rot = cam.get("rotation_euler")

    if K_list is None or loc is None or rot is None:
        return None, None

    K = np.array(K_list, dtype=np.float32)
    T_wc = blender_to_opencv_T_wc(loc, rot)

    return K, T_wc


# ──────────────────────────────────────────────────────────────────────────────
# Multi-view Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TrunkDA2MV(Dataset):
    """
    Multi-view extension of TrunkDA2.

    Builds an index grouped by (bark, tree, shot) so that each primary sample
    can be paired with a nearby secondary view from a different camera.

    Nearby pairing: at init, camera locations are loaded from annotation JSONs
    and within each scene group the cameras are sorted by Euclidean distance.
    At sample time we pick from the K nearest neighbours (default K=5) with
    inverse-distance weighting, so closer cameras are chosen more often.
    This ensures high pixel overlap for the multi-view warp loss.

    Only rows with a valid ann_path are included (camera pose is required for
    the multi-view consistency loss).

    Args:
        manifest_path : path to train_manifest.csv or val_manifest.csv
        mode          : 'train' or 'val'
        size          : (height, width) tuple, default (518, 518)
        max_neighbours: how many nearest cameras to sample from (default 5)
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        manifest_path: str,
        mode: str = "train",
        size: tuple = (518, 518),
        max_neighbours: int = 5,
        random_swap_pair: bool = False,
    ) -> None:
        assert mode in ("train", "val")
        self.mode = mode
        self._random_swap_pair = random_swap_pair

        # Load manifest — keep only rows with ann_path
        all_rows: list[dict] = []
        with open(manifest_path, newline="") as fh:
            for row in csv.DictReader(fh):
                all_rows.append(row)

        # Filter to rows that have annotation files
        self.rows = [r for r in all_rows if r.get("ann_path", "")]

        # Group row indices by (bark, tree, shot) for multi-view pairing
        self._scene_groups: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.rows):
            key = f"{r['bark']}_{r['tree']}_{r['shot']}"
            self._scene_groups[key].append(i)

        # Map each row to its scene key
        self._row_scene_key = []
        for r in self.rows:
            self._row_scene_key.append(f"{r['bark']}_{r['tree']}_{r['shot']}")

        # ── Precompute nearest-neighbour pairs within each scene group ────────
        # For each row index, store a list of (neighbour_idx, distance) sorted
        # by ascending distance.  At sample time we pick from the top-K.
        self._nearby: dict[int, list[tuple[int, float]]] = {}
        self._max_neighbours = max_neighbours

        # Load camera locations once (only need xyz position, not full pose)
        cam_locs: dict[int, np.ndarray] = {}
        for i, r in enumerate(self.rows):
            ann_path = r.get("ann_path", "")
            if not ann_path:
                continue
            try:
                with open(ann_path) as f:
                    ann = json.load(f)
                loc = ann["camera"]["location"]
                cam_locs[i] = np.array(loc, dtype=np.float64)
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                pass

        # For each scene group, compute pairwise distances and rank neighbours
        n_no_neighbours = 0
        for key, group in self._scene_groups.items():
            # Only consider members that have a camera location
            with_loc = [g for g in group if g in cam_locs]
            for idx in group:
                if idx not in cam_locs or len(with_loc) < 2:
                    n_no_neighbours += 1
                    continue
                dists = []
                for other in with_loc:
                    if other == idx:
                        continue
                    d = np.linalg.norm(cam_locs[idx] - cam_locs[other])
                    dists.append((other, float(d)))
                dists.sort(key=lambda x: x[1])
                self._nearby[idx] = dists

        n_with = len(self._nearby)
        print(f"  nearby-pairing: {n_with:,} rows with neighbours, "
              f"{n_no_neighbours:,} without")

        # Curriculum support
        first = self.rows[0] if self.rows else {}
        has_set_id = "set_id" in first

        if has_set_id:
            self._base_idxs = np.array(
                [i for i, r in enumerate(self.rows) if r["set_id"] == "base"],
                dtype=np.int64,
            )
            self._cam_idxs = np.array(
                [i for i, r in enumerate(self.rows) if r["set_id"] != "base"],
                dtype=np.int64,
            )
        else:
            self._base_idxs = np.arange(len(self.rows), dtype=np.int64)
            self._cam_idxs = np.array([], dtype=np.int64)

        self._active = np.arange(len(self.rows), dtype=np.int64)

        # Transform pipeline (same as TrunkDA2)
        resize = Resize(
            width=size[1], height=size[0],
            keep_aspect_ratio=True, ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )
        normalize = NormalizeImage(mean=self.MEAN, std=self.STD)
        prepare = PrepareForNet()

        if mode == "train":
            self._resize_norm_prep = lambda s: prepare(normalize(resize(s)))
            self._crop = Crop(size)
        else:
            self._resize_norm_prep = lambda s: prepare(normalize(resize(s)))
            self._crop = None

    def set_curriculum_epoch(self, epoch: int) -> None:
        """Same curriculum logic as TrunkDA2."""
        n_total = len(self.rows)
        if epoch < 2:
            idx = self._base_idxs.copy()
        else:
            n_base = max(1, round(n_total * 0.30))
            n_cam = n_total - n_base
            base_s = np.random.choice(self._base_idxs, size=n_base, replace=True)
            cam_s = np.random.choice(self._cam_idxs, size=n_cam, replace=True)
            idx = np.concatenate([base_s, cam_s])
        np.random.shuffle(idx)
        self._active = idx

    def __len__(self) -> int:
        return len(self._active)

    def _load_sample(self, row: dict):
        """Load a single (image, depth, mask) sample without transforms."""
        bgr = cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {row['rgb_path']}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = np.load(row["depth_path"]).astype(np.float32)

        mask_raw = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            # Mask file missing (common for cam* views) — treat whole image as valid
            H, W = image.shape[:2]
            mask = np.ones((H, W), dtype=np.float32)
        else:
            mask = (mask_raw > 0).astype(np.float32)

        return {"image": image, "depth": depth, "mask": mask}

    def __getitem__(self, idx: int) -> dict:
        real_idx = int(self._active[idx])
        row1 = self.rows[real_idx]

        # ── Pick secondary view ───────────────────────────────────────────
        # STEREO MODE: manifest has explicit pair_* columns (L/R optical-flow
        # images from the same camera position).  Use them directly.
        #
        # PROXIMITY MODE: no explicit pair — pick the nearest-neighbour camera
        # from the same (bark, tree, shot) scene group.
        if row1.get("pair_rgb_path", ""):
            # Stereo pair: build a synthetic row2 from the pair_* columns.
            # With random_swap_pair, randomly choose which side (L or R) is
            # primary so the model sees both directions during training.
            row2 = {
                "rgb_path":   row1["pair_rgb_path"],
                "depth_path": row1["pair_depth_path"],
                "mask_path":  row1.get("pair_mask_path", ""),
                "ann_path":   row1.get("pair_ann_path", ""),
            }
            if self._random_swap_pair and random.random() < 0.5:
                row1, row2 = row2, row1
        elif real_idx in self._nearby:
            neighbours = self._nearby[real_idx]
            top_k = neighbours[: self._max_neighbours]
            dists = np.array([d for _, d in top_k], dtype=np.float64)
            weights = 1.0 / (dists + 1e-6)
            weights /= weights.sum()
            chosen = np.random.choice(len(top_k), p=weights)
            row2 = self.rows[top_k[chosen][0]]
        else:
            # No neighbour info — fall back to random from scene group
            scene_key = self._row_scene_key[real_idx]
            group = self._scene_groups[scene_key]
            candidates = [g for g in group if g != real_idx]
            row2 = self.rows[random.choice(candidates)] if candidates else row1

        # Load both samples — record original resolution before any transforms
        s1 = self._load_sample(row1)
        s2 = self._load_sample(row2)
        orig_H1, orig_W1 = s1["image"].shape[:2]
        orig_H2, orig_W2 = s2["image"].shape[:2]

        # Apply resize + normalize + prepare (shared transform, no crop yet)
        s1 = self._resize_norm_prep(s1)
        s2 = self._resize_norm_prep(s2)

        # Resized dimensions — images are now (C, H_r, W_r)
        _, rH1, rW1 = s1["image"].shape
        _, rH2, rW2 = s2["image"].shape

        # For training, apply the SAME random crop to both views so pixels
        # correspond after reprojection
        h_start, w_start = 0, 0  # default when no crop (val mode)
        if self._crop is not None:
            h, w = s1["image"].shape[-2:]
            crop_h, crop_w = self._crop.size
            h_start = np.random.randint(0, h - crop_h + 1)
            w_start = np.random.randint(0, w - crop_w + 1)
            h_end = h_start + crop_h
            w_end = w_start + crop_w

            for s in (s1, s2):
                s["image"] = s["image"][:, h_start:h_end, w_start:w_end]
                s["depth"] = s["depth"][h_start:h_end, w_start:w_end]
                s["mask"] = s["mask"][h_start:h_end, w_start:w_end]

        # Load camera parameters
        K1, T1_wc = load_camera_from_ann(row1.get("ann_path", ""))
        K2, T2_wc = load_camera_from_ann(row2.get("ann_path", ""))

        # Fallback: identity if annotation missing
        if K1 is None:
            K1 = np.eye(3, dtype=np.float32)
            T1_wc = np.eye(4, dtype=np.float32)
        if K2 is None:
            K2 = np.eye(3, dtype=np.float32)
            T2_wc = np.eye(4, dtype=np.float32)

        # Rescale K from original resolution to resized resolution
        K1 = K1.copy()
        K1[0, 0] *= rW1 / orig_W1; K1[0, 2] *= rW1 / orig_W1  # fx, cx
        K1[1, 1] *= rH1 / orig_H1; K1[1, 2] *= rH1 / orig_H1  # fy, cy
        K2 = K2.copy()
        K2[0, 0] *= rW2 / orig_W2; K2[0, 2] *= rW2 / orig_W2
        K2[1, 1] *= rH2 / orig_H2; K2[1, 2] *= rH2 / orig_H2

        # Shift principal point for crop offset (same crop applied to both views)
        K1[0, 2] -= w_start; K1[1, 2] -= h_start
        K2[0, 2] -= w_start; K2[1, 2] -= h_start

        # Tight bounding-box mask for each view (used when --eval-box-mask is set)
        def _box_mask(m: np.ndarray) -> np.ndarray:
            ys, xs = np.where(m > 0.5)
            if len(ys):
                bm = np.zeros_like(m)
                bm[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1.0
                return bm
            return np.zeros_like(m)

        return {
            "image":      torch.from_numpy(s1["image"]),
            "depth":      torch.from_numpy(s1["depth"]),
            "valid_mask": torch.from_numpy(s1["mask"]),
            "box_mask":   torch.from_numpy(_box_mask(s1["mask"])),
            "image2":      torch.from_numpy(s2["image"]),
            "depth2":      torch.from_numpy(s2["depth"]),
            "valid_mask2": torch.from_numpy(s2["mask"]),
            "K1":    torch.from_numpy(K1),
            "K2":    torch.from_numpy(K2),
            "T1_wc": torch.from_numpy(T1_wc),
            "T2_wc": torch.from_numpy(T2_wc),
        }
