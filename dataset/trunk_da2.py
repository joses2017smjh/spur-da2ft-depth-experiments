"""
dataset/trunk_da2.py
────────────────────
Manifest-based dataset that mirrors the official Depth Anything V2
metric_depth dataset interface (Hypersim / VKITTI2).

Transform pipeline is a verbatim copy of:
  metric_depth/dataset/transform.py
  github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth

Key differences from Hypersim:
  - Loads from a manifest CSV (rgb_path, depth_path, mask_path columns)
  - depth is Blender camera-space Z in metres (float32 .npy)
  - valid_mask is the tree/trunk binary mask (not NaN-based)
  - Depth-range masking is NOT applied here — the training loop does it
    exactly as the official train.py:
        (valid_mask == 1) & (depth >= min_depth) & (depth <= max_depth)

Sample dict returned (identical key set to Hypersim):
    image      : FloatTensor (3, H, W)  — ImageNet-normalised
    depth      : FloatTensor (H, W)     — metric depth in metres
    valid_mask : FloatTensor (H, W)     — {0.0, 1.0}  trunk pixels
"""

import csv

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Transforms — verbatim copy of metric_depth/dataset/transform.py
# (only keys used: "image", "depth", "mask")
# ──────────────────────────────────────────────────────────────────────────────

class Resize:
    """Resize sample to given size (width, height).

    Verbatim from metric_depth/dataset/transform.py.
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        self.__width  = width
        self.__height = height
        self.__resize_target             = resize_target
        self.__keep_aspect_ratio         = keep_aspect_ratio
        self.__multiple_of               = ensure_multiple_of
        self.__resize_method             = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)
        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)
        return y

    def get_size(self, width, height):
        scale_height = self.__height / height
        scale_width  = self.__width  / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width  = self.constrain_to_multiple_of(scale_width  * width,  min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width  = self.constrain_to_multiple_of(scale_width  * width,  max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width  = self.constrain_to_multiple_of(scale_width  * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(
            sample["image"].shape[1], sample["image"].shape[0]
        )

        # Resize image (HxWxC float32)
        sample["image"] = cv2.resize(
            sample["image"],
            (width, height),
            interpolation=self.__image_interpolation_method,
        )

        if self.__resize_target:
            if "depth" in sample:
                sample["depth"] = cv2.resize(
                    sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST
                )
            if "mask" in sample:
                sample["mask"] = cv2.resize(
                    sample["mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            if "box_mask" in sample:
                sample["box_mask"] = cv2.resize(
                    sample["box_mask"].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

        return sample


class NormalizeImage:
    """Normalise image by given mean and std.

    Verbatim from metric_depth/dataset/transform.py.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std  = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std
        return sample


class PrepareForNet:
    """Prepare sample for usage as network input.

    Transposes image HxWxC → CxHxW and ensures contiguous float32.
    Verbatim from metric_depth/dataset/transform.py.
    """

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])

        if "box_mask" in sample:
            sample["box_mask"] = sample["box_mask"].astype(np.float32)
            sample["box_mask"] = np.ascontiguousarray(sample["box_mask"])

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)

        return sample


class Crop:
    """Random crop applied only during training.

    Operates on CxHxW format (i.e. after PrepareForNet).
    Verbatim from metric_depth/dataset/transform.py.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample["image"].shape[-2:]
        assert h >= self.size[0] and w >= self.size[1], (
            f"Crop size {self.size} larger than image {h}×{w} after Resize. "
            "This should not happen with lower_bound resize."
        )
        h_start = np.random.randint(0, h - self.size[0] + 1)
        w_start = np.random.randint(0, w - self.size[1] + 1)
        h_end   = h_start + self.size[0]
        w_end   = w_start + self.size[1]

        sample["image"] = sample["image"][:, h_start:h_end, w_start:w_end]

        if "depth" in sample:
            sample["depth"] = sample["depth"][h_start:h_end, w_start:w_end]

        if "mask" in sample:
            sample["mask"] = sample["mask"][h_start:h_end, w_start:w_end]

        if "box_mask" in sample:
            sample["box_mask"] = sample["box_mask"][h_start:h_end, w_start:w_end]

        return sample


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TrunkDA2(Dataset):
    """
    Manifest-based dataset that mirrors the Hypersim dataset interface used
    by the official Depth Anything V2 metric training script.

    Args:
        manifest_path : path to train_manifest.csv or val_manifest.csv
                        Required CSV columns: rgb_path, depth_path, mask_path
        mode          : 'train' applies RandomCrop; 'val' does not.
                        Mirrors official Hypersim dataset mode argument.
        size          : (height, width) tuple, default (518, 518).

    Transforms (official order from metric_depth/dataset/hypersim.py):
        1. Resize(width, height, keep_aspect_ratio=True,
                  ensure_multiple_of=14, resize_method="lower_bound")
              → image is ≥ size, aspect ratio preserved, snapped to 14×n
        2. NormalizeImage(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        3. PrepareForNet()   → transposes image to CxHxW
        4. Crop(size)        → train only; random 518×518 window

    Returned dict keys (identical to official Hypersim __getitem__):
        image      : FloatTensor (3, H, W)  — ImageNet-normalised
        depth      : FloatTensor (H, W)     — metric Z in metres
        valid_mask : FloatTensor (H, W)     — {0.0, 1.0} trunk mask

    NOTE: valid_mask is the raw trunk mask only (no depth-range masking).
          The training loop applies the official combined mask:
              (valid_mask == 1) & (depth >= min_depth) & (depth <= max_depth)
    """

    # ImageNet statistics — same constants used throughout official codebase
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        manifest_path: str,
        mode: str = "train",
        size: tuple = (518, 518),
        box_dilation_px: int = 8,
        global_box_mask_path: str = "",
    ) -> None:
        assert mode in ("train", "val"), f"mode must be 'train' or 'val', got {mode!r}"
        self.mode = mode
        # box_mask source: either a global file (applied to every sample) or a
        # per-sample dilation of the trunk silhouette. If global_box_mask_path is
        # set, it overrides dilation.
        self.box_dilation_px = int(box_dilation_px)
        self.global_box_mask_path = global_box_mask_path or ""
        self._global_box_mask = None
        if self.global_box_mask_path:
            gbm = cv2.imread(self.global_box_mask_path, cv2.IMREAD_GRAYSCALE)
            if gbm is None:
                raise FileNotFoundError(
                    f"global_box_mask_path={self.global_box_mask_path!r} not readable")
            self._global_box_mask = (gbm > 0).astype(np.float32)  # (H, W) {0,1}

        # Load manifest rows
        self.rows: list[dict] = []
        with open(manifest_path, newline="") as fh:
            for row in csv.DictReader(fh):
                self.rows.append(row)

        # ── Curriculum index sets ─────────────────────────────────────────────
        # Matches the original TrunkManifest curriculum logic exactly.
        # set_id == "base"  → base images (close-up, well-lit)
        # set_id != "base"  → camera-variation images
        # If the CSV has no set_id column, all rows are treated as base.
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
            self._cam_idxs  = np.array([], dtype=np.int64)

        # Default active view: all rows visible.
        # For val mode this never changes.
        # For train mode, call set_curriculum_epoch(epoch) each epoch.
        self._active = np.arange(len(self.rows), dtype=np.int64)

        # Build transform pipeline — exact mirror of Hypersim's transform setup.
        # Train: Resize → NormalizeImage → PrepareForNet → Crop
        # Val:   Resize → NormalizeImage → PrepareForNet
        #
        # Resize uses lower_bound + keep_aspect_ratio + ensure_multiple_of=14
        # so that output is ALWAYS ≥ 518×518 and divisible by the patch size.
        # RandomCrop then selects a 518×518 window for training batches.
        resize = Resize(
            width=size[1],
            height=size[0],
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        )
        normalize = NormalizeImage(mean=self.MEAN, std=self.STD)
        prepare   = PrepareForNet()

        if mode == "train":
            self.transform = lambda s: Crop(size)(prepare(normalize(resize(s))))
        else:
            self.transform = lambda s: prepare(normalize(resize(s)))

    # ── Curriculum ────────────────────────────────────────────────────────────

    def set_curriculum_epoch(self, epoch: int) -> None:
        """Update which rows are visible for this epoch.

        Call once per epoch BEFORE constructing the DataLoader.

        epoch 0–1  →  base images only
        epoch >= 2 →  30 % base / 70 % cam, total = full dataset size
                       (sampling with replacement so both pools are always used)

        Mirrors the original TrunkManifest.set_curriculum_epoch() exactly.
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

    def __len__(self) -> int:
        return len(self._active)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[self._active[idx]]

        # ── RGB image: BGR → RGB, float32 [0, 1] ─────────────────────────────
        # Mirrors Hypersim: reads as float32 HxWxC normalised to [0,1] before
        # NormalizeImage applies ImageNet mean/std subtraction.
        bgr = cv2.imread(row["rgb_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {row['rgb_path']}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # ── Depth: Blender camera-space Z in metres ───────────────────────────
        depth = np.load(row["depth_path"]).astype(np.float32)

        # ── Trunk mask: binary, normalised to {0.0, 1.0} float32 ─────────────
        # Stored as {0.0, 1.0} so that (valid_mask == 1) in the training loop
        # works exactly as in the official train.py.
        mask_raw = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            # Mask file missing (common for cam* views) — treat whole image as valid
            H, W = image.shape[:2]
            mask = np.ones((H, W), dtype=np.float32)
        else:
            mask = (mask_raw > 0).astype(np.float32)  # {0.0, 1.0}

        # ── Build sample dict matching official Resize/PrepareForNet key names ─
        sample = {"image": image, "depth": depth, "mask": mask}

        # If a global box mask is configured, attach it so the transform
        # pipeline (Resize + Crop) processes it identically to the trunk mask.
        if self._global_box_mask is not None:
            gbm = self._global_box_mask
            H, W = image.shape[:2]
            if gbm.shape != (H, W):
                gbm = cv2.resize(gbm, (W, H), interpolation=cv2.INTER_NEAREST)
            sample["box_mask"] = gbm.copy()

        # ── Apply transform pipeline ──────────────────────────────────────────
        sample = self.transform(sample)

        # ── Convert numpy → torch tensors ────────────────────────────────────
        trunk_mask = sample["mask"]   # (H, W) float32 {0,1}

        if self._global_box_mask is not None:
            # Global mask passed through Resize + Crop alongside trunk mask.
            box_mask = (sample["box_mask"] > 0.5).astype(np.float32)
        elif self.box_dilation_px > 0:
            # Fallback: trunk silhouette dilated by box_dilation_px pixels.
            k = 2 * self.box_dilation_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            box_mask = cv2.dilate(
                (trunk_mask > 0.5).astype(np.uint8), kernel
            ).astype(np.float32)
        else:
            box_mask = (trunk_mask > 0.5).astype(np.float32)

        return {
            "image":      torch.from_numpy(sample["image"]),       # (3, H, W) float32
            "depth":      torch.from_numpy(sample["depth"]),       # (H, W)    float32
            "valid_mask": torch.from_numpy(trunk_mask),            # (H, W)    float32 {0,1}
            "box_mask":   torch.from_numpy(box_mask),              # (H, W)    float32 {0,1}
        }
