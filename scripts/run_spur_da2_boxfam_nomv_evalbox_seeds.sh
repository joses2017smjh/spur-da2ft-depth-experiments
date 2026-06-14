#!/bin/bash
#SBATCH --job-name=spur_da2_boxfam_nomv_evalbox_v3
#SBATCH --array=1-5
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.err

# DA2 fine-tune on full_spur, box-family set_ids, no MV loss.
# Identical to spur_da2_boxfam_nomv_v3 but with --eval-box-mask so val loop
# reports both trunk-mask RMSE/SiLog AND box-mask RMSE/SiLog each epoch.
# 5 seeds via SLURM array, 80 epochs, patience=10, 80/20 tree-level split.

set -euo pipefail

mkdir -p /tmp/sanchej7_tmp && export TMPDIR=/tmp/sanchej7_tmp

source /nfs/hpc/share/sanchej7/miniforge3_fixed/etc/profile.d/conda.sh
conda activate /nfs/stak/users/sanchej7/miniforge3/envs/depth-env

export COMPUTER_VISION_ROOT=/nfs/stak/users/sanchej7/hpc-share/Computer_Vision
cd "$COMPUTER_VISION_ROOT"

DA2_ROOT="$COMPUTER_VISION_ROOT/depth-anything-v2"
export PYTHONPATH="${DA2_ROOT}:${DA2_ROOT}/metric_depth${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME=/nfs/hpc/share/sanchej7/.cache/huggingface
export TORCH_HOME=/nfs/hpc/share/sanchej7/.cache/torch
export WANDB_API_KEY="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"
export WANDB_DIR=/tmp/sanchej7_tmp
export WANDB_CACHE_DIR=/tmp/sanchej7_tmp/wandb_cache
mkdir -p "$WANDB_CACHE_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEED=$SLURM_ARRAY_TASK_ID
export MANIFEST_DIR="/nfs/hpc/share/sanchej7/Computer_Vision/manifests/spur_boxfam_v3_seed${SEED}"
CKPT_DIR="/nfs/hpc/share/sanchej7/Computer_Vision/checkpoints/spur_da2_boxfam_nomv_evalbox_v3_seed${SEED}"
PRETRAINED="/nfs/hpc/share/sanchej7/Computer_Vision/da2_weights/depth_anything_v2_vitl.pth"
export SPUR_ROOT="/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur"

mkdir -p "$MANIFEST_DIR" "$CKPT_DIR"
mkdir -p /nfs/hpc/share/sanchej7/Computer_Vision/logs

echo "===== JOB START  seed=${SEED}  array_id=${SLURM_ARRAY_TASK_ID} ====="
date; hostname; nvidia-smi; python --version

# ── Build box-family spur manifests (shared dir with spur_da2_boxfam_nomv_v3) ─
python3 - <<'PYEOF'
import os, csv, random
from pathlib import Path
import cv2

seed      = int(os.environ["SLURM_ARRAY_TASK_ID"])
mdir      = Path(os.environ["MANIFEST_DIR"])
spur_root = Path(os.environ["SPUR_ROOT"])
random.seed(seed)

BOX_FAMILY = {
    "box",
    "box_cam1", "box_cam2", "box_cam3", "box_cam4",
    "box_cam5", "box_cam6", "box_cam7", "box_cam8",
}
BARK  = "bark_brown_02"
SHOTS = [f"shot{i:02d}" for i in range(1, 7)]

of_root   = spur_root / "Optical_flow" / BARK
dep_root  = spur_root / "depth"        / BARK
mask_root = spur_root / "mask"         / BARK
ann_root  = spur_root / "ann"          / BARK

def mask_ok(path):
    """Return True iff mask exists and contains at least one non-zero pixel."""
    if not path:
        return False
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return m is not None and bool(m.max() > 0)

rows = []
for tree_dir in sorted(of_root.iterdir()):
    tree = tree_dir.name
    for sid in sorted(BOX_FAMILY):
        of_dir   = of_root   / tree / sid
        dep_dir  = dep_root  / tree / sid
        mask_dir = mask_root / tree / sid
        ann_dir  = ann_root  / tree / sid
        if not of_dir.exists():
            continue
        for shot in SHOTS:
            l_png  = of_dir   / f"{tree}_{shot}_l.png"
            r_png  = of_dir   / f"{tree}_{shot}_r.png"
            l_dep  = dep_dir  / f"{tree}_{shot}_l.npy"
            r_dep  = dep_dir  / f"{tree}_{shot}_r.npy"
            l_mask = mask_dir / f"{tree}_{shot}_l.png"
            r_mask = mask_dir / f"{tree}_{shot}_r.png"
            l_ann  = ann_dir  / f"{tree}_{shot}_l.json"
            r_ann  = ann_dir  / f"{tree}_{shot}_r.json"
            if not (l_png.exists() and r_png.exists() and
                    l_dep.exists() and r_dep.exists() and
                    l_ann.exists() and r_ann.exists()):
                continue
            if not (mask_ok(str(l_mask)) and mask_ok(str(r_mask))):
                continue
            rows.append({
                "bark": BARK, "tree": tree, "set_id": sid, "shot": shot,
                "rgb_path":        str(l_png),
                "depth_path":      str(l_dep),
                "mask_path":       str(l_mask),
                "ann_path":        str(l_ann),
                "pair_rgb_path":   str(r_png),
                "pair_depth_path": str(r_dep),
                "pair_mask_path":  str(r_mask),
                "pair_ann_path":   str(r_ann),
            })

all_trees = sorted({r["tree"] for r in rows})
random.shuffle(all_trees)
n_train = max(1, round(len(all_trees) * 0.8))
train_trees = set(all_trees[:n_train])
val_trees   = set(all_trees[n_train:])
assert not train_trees & val_trees

fields = ["bark", "tree", "set_id", "shot",
          "rgb_path", "depth_path", "mask_path", "ann_path",
          "pair_rgb_path", "pair_depth_path", "pair_mask_path", "pair_ann_path"]

for name, trees in [("train", train_trees), ("val", val_trees)]:
    sel = sorted([r for r in rows if r["tree"] in trees],
                 key=lambda r: (r["tree"], r["set_id"], r["shot"]))
    p = mdir / f"spur_boxfam_{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(sel)
    print(f"  {p.name}: {len(sel):,} rows, {len(trees)} trees")

print(f"train trees={sorted(train_trees)}  val trees={sorted(val_trees)}")
PYEOF

echo "Manifests ready (seed=${SEED})."

python train_depth_da2_mv.py \
    --encoder           vitl \
    --pretrained-from   "$PRETRAINED" \
    --train-manifest    "$MANIFEST_DIR/spur_boxfam_train.csv" \
    --val-manifest      "$MANIFEST_DIR/spur_boxfam_val.csv" \
    --save-path         "$CKPT_DIR" \
    --epochs            80 \
    --bs                2 \
    --lr                0.000005 \
    --lambda-mv         0.0 \
    --mv-warmup-epochs  0 \
    --patience          10 \
    --save-every        0 \
    --num-workers       4 \
    --seed              $SEED \
    --eval-box-mask \
    --wandb-project     "bark02-full-spur-ablation" \
    --wandb-group       "spur_da2_boxfam_nomv_evalbox_v3" \
    --wandb-run-name    "spur_da2_boxfam_nomv_evalbox_v3_seed${SEED}"

echo "===== DONE  seed=${SEED} ====="
date
