"""
exr_to_npy.py — Convert Blender depth EXR files to .npy (float32).

Usage:
    # Test mode (default): process only the first EXR found
    python exr_to_npy.py

    # Process ALL EXR files in the directory tree
    python exr_to_npy.py --all

    # Custom input directory
    python exr_to_npy.py --all --input-dir /path/to/depth/folder
"""

import os
import sys
import argparse
import numpy as np
import OpenEXR
import Imath

# Default depth directory
DEFAULT_DEPTH_DIR = (
    "/nfs/stak/users/sanchej7/hpc-share/Computer_Vision"
    "/Data/trunk_spurs/depth/bark_brown/lpy_envy_00000"
)


def load_exr(path: str) -> np.ndarray:
    """Load depth from Blender EXR (ViewLayer.Depth.Z)."""
    exr = OpenEXR.InputFile(path)
    hdr = exr.header()
    dw = hdr["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    Z = np.frombuffer(exr.channel("ViewLayer.Depth.Z", pt),
                      dtype=np.float32).reshape(h, w)
    exr.close()
    return Z


def convert_exr_to_npy(exr_path: str) -> str:
    """Convert a single EXR to .npy (float32). Returns output path."""
    depth = load_exr(exr_path)
    npy_path = os.path.splitext(exr_path)[0] + ".npy"
    np.save(npy_path, depth.astype(np.float32))
    return npy_path


def find_exr_files(root_dir: str):
    """Recursively find all .exr files under root_dir."""
    exr_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if fname.lower().endswith(".exr"):
                exr_files.append(os.path.join(dirpath, fname))
    return exr_files


def main():
    parser = argparse.ArgumentParser(description="Convert depth EXR files to .npy")
    parser.add_argument("--all", action="store_true",
                        help="Process all EXR files (default: test mode, first file only)")
    parser.add_argument("--input-dir", default=DEFAULT_DEPTH_DIR,
                        help=f"Root directory to search for EXR files (default: {DEFAULT_DEPTH_DIR})")
    args = parser.parse_args()

    depth_dir = args.input_dir
    if not os.path.isdir(depth_dir):
        print(f"ERROR: directory not found: {depth_dir}")
        sys.exit(1)

    exr_files = find_exr_files(depth_dir)
    if not exr_files:
        print(f"No .exr files found in {depth_dir}")
        sys.exit(0)

    if not args.all:
        # Test mode: only process the first file
        exr_files = exr_files[:1]
        print(f"TEST MODE: processing 1 of {len(find_exr_files(depth_dir))} EXR files "
              f"(use --all to process everything)\n")

    total = len(exr_files)
    for i, exr_path in enumerate(exr_files, start=1):
        exr_size = os.path.getsize(exr_path)
        npy_path = convert_exr_to_npy(exr_path)
        npy_size = os.path.getsize(npy_path)
        ratio = npy_size / exr_size * 100 if exr_size > 0 else 0
        os.remove(exr_path)
        print(f"[{i}/{total}] {os.path.basename(exr_path)} -> {os.path.basename(npy_path)}  "
              f"({exr_size / 1024:.0f} KB -> {npy_size / 1024:.0f} KB, {ratio:.1f}%) [exr deleted]")

    print(f"\nDone: converted {total} file(s)")


if __name__ == "__main__":
    main()
