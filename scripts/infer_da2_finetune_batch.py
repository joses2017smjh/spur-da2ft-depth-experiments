#!/usr/bin/env python3
"""
infer_da2_finetune_batch.py — batch DA2 (fine-tuned) depth inference.

Reads a tab-separated frame list (one "rgb_path<TAB>output_npy_path" per line),
runs the fine-tuned Depth-Anything-v2 ViT-L model on each RGB image, and saves
the predicted metric depth as a float32 .npy at output_npy_path (parent dirs
created as needed). Existing outputs are skipped unless --overwrite.

Usage:
    python infer_da2_finetune_batch.py \
        --frame-list  manifests/da2finetune_frames.txt \
        --ckpt        checkpoints/full_spur_2tex_all_3view_seed1/best.pth \
        --da2-root    depth-anything-v2/metric_depth \
        --max-depth   20.0
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-list", required=True,
                    help="TSV file: each line 'rgb_path<TAB>output_npy_path'.")
    ap.add_argument("--ckpt", required=True,
                    help="Fine-tuned DA2 checkpoint (.pth with a 'model' key).")
    ap.add_argument("--da2-root", required=True,
                    help="Path to depth-anything-v2/metric_depth (for imports).")
    ap.add_argument("--encoder", default="vitl")
    ap.add_argument("--max-depth", type=float, default=20.0,
                    help="Must match the value used at fine-tune time.")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.da2_root)
    from depth_anything_v2.dpt import DepthAnythingV2

    with open(args.frame_list) as fh:
        pairs = [ln.rstrip("\n").split("\t") for ln in fh if ln.strip()]
    pairs = [(rg, out) for rg, out in pairs]
    print(f"Frames in list: {len(pairs):,}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  loading model from {args.ckpt}", flush=True)
    model = DepthAnythingV2(
        encoder=args.encoder,
        features=256,
        out_channels=[256, 512, 1024, 1024],
        max_depth=args.max_depth,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device).eval()
    print("model loaded", flush=True)

    n_done = n_skip = n_miss = 0
    t0 = time.time()
    for i, (rgb_path, out_path) in enumerate(pairs):
        if (not args.overwrite) and os.path.exists(out_path):
            n_skip += 1
            continue
        img = cv2.imread(rgb_path)
        if img is None:
            print(f"[MISS] cannot read {rgb_path}", flush=True)
            n_miss += 1
            continue
        with torch.no_grad():
            pred = model.infer_image(img, input_size=args.input_size)  # HxW float32, metres
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # atomic write
        tmp = out_path + ".tmp.npy"
        np.save(tmp, pred.astype(np.float32))
        os.replace(tmp, out_path)
        n_done += 1
        if n_done % 100 == 0:
            rate = n_done / max(time.time() - t0, 1e-6)
            print(f"  {i+1}/{len(pairs)}  done={n_done} skip={n_skip} miss={n_miss}  "
                  f"({rate:.1f} img/s)", flush=True)

    dt = time.time() - t0
    print(f"\nFINISHED: done={n_done} skip={n_skip} miss={n_miss}  in {dt/60:.1f} min",
          flush=True)
    if n_miss > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
