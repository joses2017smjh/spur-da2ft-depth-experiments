#!/bin/bash
#SBATCH --job-name=spur_dino_stereo_boxfam
#SBATCH --array=1-5
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.err

# DINOv2 + DSB stereo ablation — full_spur/bark_brown_02, box-family, 5 seeds.
#
# Architecture: MVStereoDINOUNet(n_views=2, use_pose=False)
#   Frozen DINOv2 ViT-L encoder + DepthSideBranch + cross-view bottleneck fusion.
# Input:  RGB + PRO depth pair (L, R)
# Output: Refined depth per frame
# Loss:   SiLog on trunk-masked GT depth (both views); MV consistency OFF (lambda=0)
#
# PRO depth normalisation (already applied in TrunkStereoMVPDataset._load_pro_depth):
#   depth_cal = _PRO_ALPHA * depth_raw + _PRO_BETA
#   _PRO_ALPHA = -0.06610793956568871   (global_alpha, best entry in benchmark_pro_sweep.json)
#   _PRO_BETA  =  1.555980697834118     (global_beta,  erode_r=10, winsor_pct=[0,100],
#                                        fit_space=depth, global RMSE=0.0826 m)
#
# Dataset:  full_spur/bark_brown_02, box-family {box, box_cam1..box_cam8}
#           8 train trees / 2 val trees sampled from pre-split source manifests (no leakage)
# Hyper-params mirror the full_trunk/bark_brown_02 baseline (job 20280791, 0.361 ± 0.016 m RMSE)

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

export SRC_STEREO_TRAIN="/nfs/hpc/share/sanchej7/Computer_Vision/manifests/source/spur_stereo_train_manifest.csv"
export SRC_STEREO_VAL="/nfs/hpc/share/sanchej7/Computer_Vision/manifests/source/spur_stereo_val_manifest.csv"
export MANIFEST_DIR="/nfs/hpc/share/sanchej7/Computer_Vision/manifests/spur_dino_stereo_boxfam_seed${SEED}"
CKPT_ROOT="/nfs/hpc/share/sanchej7/Computer_Vision/checkpoints/spur_dino_stereo_boxfam_seed${SEED}"

mkdir -p "$MANIFEST_DIR"
mkdir -p "$CKPT_ROOT/exp_dino_stereo_boxfam"
mkdir -p /nfs/hpc/share/sanchej7/Computer_Vision/logs

echo "===== JOB START  seed=${SEED}  array_id=${SLURM_ARRAY_TASK_ID} ====="
date; hostname; nvidia-smi; python --version

# Build box-family manifests if not already present
if [ ! -f "$MANIFEST_DIR/stereo_train_boxfam.csv" ]; then
echo "Building box-family manifests for seed=${SEED}..."
python3 - <<PYEOF
import csv, random, os
import cv2

seed = int(os.environ["SLURM_ARRAY_TASK_ID"])
random.seed(seed)

SRC_TRAIN    = os.environ["SRC_STEREO_TRAIN"]
SRC_VAL      = os.environ["SRC_STEREO_VAL"]
MANIFEST_DIR = os.environ["MANIFEST_DIR"]

BOX_FAMILY = {
    "box",
    "box_cam1", "box_cam2", "box_cam3", "box_cam4",
    "box_cam5", "box_cam6", "box_cam7", "box_cam8",
}

def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def mask_ok(path):
    if not path:
        return False
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return m is not None and bool(m.max() > 0)

train_rows = [r for r in read_csv(SRC_TRAIN) if r["set_id"] in BOX_FAMILY]
val_rows   = [r for r in read_csv(SRC_VAL)   if r["set_id"] in BOX_FAMILY]

train_trees = sorted({r["tree"] for r in train_rows})
val_trees   = sorted({r["tree"] for r in val_rows})
assert not set(train_trees) & set(val_trees), "Source manifest has leakage!"

sampled_train = set(random.sample(train_trees, min(8, len(train_trees))))
sampled_val   = set(random.sample(val_trees,   min(2, len(val_trees))))
assert not sampled_train & sampled_val

print(f"Seed {seed}: train trees = {sorted(sampled_train)}")
print(f"Seed {seed}: val   trees = {sorted(sampled_val)}")

for name, rows, trees in [("train", train_rows, sampled_train), ("val", val_rows, sampled_val)]:
    sel = sorted([r for r in rows if r["tree"] in trees
                  and mask_ok(r["mask_path"]) and mask_ok(r["pair_mask_path"])],
                 key=lambda r: (r["tree"], r["set_id"], r["shot"]))
    p = f"{MANIFEST_DIR}/stereo_{name}_boxfam.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sel[0].keys()))
        w.writeheader(); w.writerows(sel)
    print(f"  {p}: {len(sel):,} rows, {len(trees)} trees")

print(f"Done (seed={seed}).")
PYEOF
fi
echo "Manifests ready (seed=${SEED})."

echo ""
echo "=== [seed=${SEED}] DINOv2 RGB+depth stereo, box-family, full_spur/bark_brown_02 ==="
python MVP_MODEL/train_mvp_stereo_dino.py \
    --n_views          2 \
    --no_pose \
    --train_manifest   "$MANIFEST_DIR/stereo_train_boxfam.csv" \
    --val_manifest     "$MANIFEST_DIR/stereo_val_boxfam.csv" \
    --H                280 \
    --W                512 \
    --epochs           80 \
    --batch_size       2 \
    --lr               3e-5 \
    --weight_decay     1e-4 \
    --patience         10 \
    --save_every       0 \
    --num_workers      4 \
    --grad_clip        1.0 \
    --min_depth        0.5 \
    --max_depth        10.0 \
    --lambda_mv        0.0 \
    --mv_warmup_epochs 2 \
    --seed             $SEED \
    --lr_warmup_epochs 5 \
    --out_dir          "$CKPT_ROOT/exp_dino_stereo_boxfam" \
    --wandb \
    --wandb_project    "bark02-full-spur-ablation" \
    --wandb_group      "spur_dino_stereo_boxfam" \
    --wandb_run_name   "spur_dino_stereo_boxfam_seed${SEED}"

echo ""
echo "===== DONE  seed=${SEED} ====="
date
