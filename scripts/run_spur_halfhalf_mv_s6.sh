#!/bin/bash
#SBATCH --job-name=spur_da2_halfhalf_mv_v2_s6
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x.err

# Rerun of spur_da2_halfhalf_mv_v2 for seed=6.
# Previous seeds 3/4/5 failed: seed 3 = wandb crash, seed 4 = disk quota on
# /nfs/hpc/share (checkpoints now on /nfs/stak instead), seed 5 = time limit
# (72h instead of 48h).

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

SEED=6
SPUR_ROOT=/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur
PRETRAINED=/nfs/hpc/share/sanchej7/Computer_Vision/da2_weights/depth_anything_v2_vitl.pth

export MANIFEST_DIR="/nfs/hpc/share/sanchej7/Computer_Vision/manifests/spur_halfhalf_seed${SEED}"
CKPT_DIR="/nfs/hpc/share/sanchej7/Computer_Vision/checkpoints/spur_da2_halfhalf_mv_v2_seed${SEED}"
mkdir -p "$MANIFEST_DIR" "$CKPT_DIR"
mkdir -p /nfs/hpc/share/sanchej7/Computer_Vision/logs

echo "===== JOB START  seed=${SEED}  spur_halfhalf_mv ====="
date; hostname; nvidia-smi; python --version

# ── Build half-half manifests from full_spur ──────────────────────────────────
python3 - <<'PYEOF'
import os, csv, random
from pathlib import Path

seed    = 6
cfg     = "halfhalf"
spur    = Path("/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur")
mdir    = Path(os.environ["MANIFEST_DIR"])
random.seed(seed)

BOX_FAMILY = {"box","box_cam1","box_cam2","box_cam3","box_cam4",
              "box_cam5","box_cam6","box_cam7","box_cam8"}
HALF_HALF  = BOX_FAMILY | {"cam1","cam2","cam3","cam4","cam5",
                            "cam6","cam7","cam8","cam9","cam10"}
set_ids = HALF_HALF

bark, bark_tag = "bark_brown_02", "bark_brown_02"
of_root   = spur / "Optical_flow" / bark
dep_root  = spur / "depth"        / bark
mask_root = spur / "mask"         / bark
ann_root  = spur / "ann"          / bark

all_trees = sorted(t.name for t in (dep_root).iterdir() if t.is_dir())
random.shuffle(all_trees)
split = int(0.8 * len(all_trees))
train_trees = all_trees[:split]
val_trees   = all_trees[split:]

SHOTS = [f"shot{i:02d}" for i in range(1, 7)]

def build_rows(trees):
    rows = []
    for tree in sorted(trees):
        for sid in sorted(set_ids):
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
                        l_dep.exists() and r_dep.exists()):
                    continue
                rows.append({
                    "bark": bark_tag, "tree": tree, "set_id": sid, "shot": shot,
                    "rgb_path":       str(l_png), "depth_path":     str(l_dep),
                    "mask_path":      str(l_mask) if l_mask.exists() else "",
                    "ann_path":       str(l_ann)  if l_ann.exists()  else "",
                    "pair_rgb_path":       str(r_png), "pair_depth_path": str(r_dep),
                    "pair_mask_path":      str(r_mask) if r_mask.exists() else "",
                    "pair_ann_path":       str(r_ann)  if r_ann.exists()  else "",
                })
    return rows

fields = ["bark","tree","set_id","shot",
          "rgb_path","depth_path","mask_path","ann_path",
          "pair_rgb_path","pair_depth_path","pair_mask_path","pair_ann_path"]

for name, rows in [("train", build_rows(train_trees)), ("val", build_rows(val_trees))]:
    p = mdir / f"spur_{cfg}_{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  {p.name}: {len(rows):,} rows, {len({r['tree'] for r in rows})} trees")

print(f"train={sorted(train_trees)}  val={sorted(val_trees)}")
PYEOF

echo ""; echo "=== [seed=${SEED}] DA2+MV, half-half spur, lambda_mv=0.1 ==="
python train_depth_da2_mv.py \
    --encoder           vitl \
    --pretrained-from   "$PRETRAINED" \
    --train-manifest    "$MANIFEST_DIR/spur_halfhalf_train.csv" \
    --val-manifest      "$MANIFEST_DIR/spur_halfhalf_val.csv" \
    --save-path         "$CKPT_DIR" \
    --epochs            80 \
    --bs                2 \
    --lr                0.000005 \
    --lambda-mv         0.1 \
    --mv-warmup-epochs  2 \
    --patience          10 \
    --save-every        0 \
    --num-workers       4 \
    --seed              $SEED \
    --eval-box-mask \
    --wandb-project     "bark02-full-spur-ablation" \
    --wandb-group       "spur_8exp_5seeds_v2" \
    --wandb-run-name    "spur_da2_halfhalf_mv_v2_seed${SEED}"

echo ""; echo "===== DONE  seed=${SEED} ====="
date
