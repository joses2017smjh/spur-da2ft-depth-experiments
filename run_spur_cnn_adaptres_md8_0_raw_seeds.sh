#!/bin/bash
#SBATCH --job-name=spur_cnn_adaptres_md8_0_raw
#SBATCH --array=1-5
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.err

# Adaptive residual depth refinement on SPUR box-family (5 seeds, 80 epochs).
#   Model:  CNN MVStereoUNet (RGB + PRO, fusion ON, pose ON)
#   PRO:    RAW PRO (no alpha,beta, --no_pro_calib)
#   Head:   2-channel adaptive-residual head, max_delta=8.0 m
#   Loss:   SiLog on masked GT depth (same as baseline)
#   Metric: trunk-mask RMSE on pred_depth (same as baseline)
#
# Formulation:
#   D_refined = D_PRO + sigmoid(gate) * max_delta * tanh(raw_delta)
# Head is zero-initialised, so at step 0: delta=0, gate=0.5, pred = D_PRO
# (a no-op warm-start). The network learns when and how much to correct.

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

DATASET="${DATASET:-full_spur}"
DSTAG="${DATASET#full_}"

VARIANT="cnn_adaptres_md8_0_raw"

export SRC_STEREO_TRAIN="/nfs/hpc/share/sanchej7/Computer_Vision/manifests/source/stereo_train_manifest.csv"
export SRC_STEREO_VAL="/nfs/hpc/share/sanchej7/Computer_Vision/manifests/source/stereo_val_manifest.csv"
export MANIFEST_DIR="/nfs/hpc/share/sanchej7/Computer_Vision/manifests/${VARIANT}_${DSTAG}_seed${SEED}"
export CKPT_ROOT="/nfs/hpc/share/sanchej7/Computer_Vision/checkpoints/${VARIANT}_${DSTAG}_seed${SEED}"

mkdir -p "$MANIFEST_DIR"
mkdir -p "$CKPT_ROOT/exp1"
mkdir -p /nfs/hpc/share/sanchej7/Computer_Vision/logs

echo "===== JOB START  variant=${VARIANT}  dataset=${DATASET}  seed=${SEED} ====="
date; hostname; nvidia-smi; python --version

# ─────────────────────────────────────────────────────────────────────────────
# Build box-family 2-view stereo manifests (same logic as baseline runs)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [seed=${SEED}] Building box-family 2-view stereo manifests ==="

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
assert not sampled_train & sampled_val, "Sampled split has leakage!"

print(f"Seed {seed}: train trees = {sorted(sampled_train)}")
print(f"Seed {seed}: val   trees = {sorted(sampled_val)}")

train_sel = [r for r in train_rows if r["tree"] in sampled_train
             and mask_ok(r["mask_path"]) and mask_ok(r["pair_mask_path"])]
train_sel = sorted(train_sel, key=lambda r: (r["tree"], r["set_id"], r["shot"]))
train_path = f"{MANIFEST_DIR}/stereo_train_boxfam.csv"
with open(train_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(train_sel[0].keys()))
    w.writeheader(); w.writerows(train_sel)
print(f"  {os.path.basename(train_path)}: {len(train_sel):,} rows")

val_sel = [r for r in val_rows if r["tree"] in sampled_val
           and mask_ok(r["mask_path"]) and mask_ok(r["pair_mask_path"])]
val_sel = sorted(val_sel, key=lambda r: (r["tree"], r["set_id"], r["shot"]))
val_path = f"{MANIFEST_DIR}/stereo_val_boxfam.csv"
with open(val_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(val_sel[0].keys()))
    w.writeheader(); w.writerows(val_sel)
print(f"  {os.path.basename(val_path)}: {len(val_sel):,} rows")

print(f"Done (seed={seed}).")
PYEOF

echo "Manifests ready for seed=${SEED}."

# ─────────────────────────────────────────────────────────────────────────────
# Training — adaptive residual head, max_delta=8.0
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [seed=${SEED}] Training ${VARIANT} ==="
python MVP_MODEL/train_mvp_stereo.py \
    --pred_mode      adaptive_residual \
    --no_pose \
    --max_delta      8.0 \
    --no_pro_calib \
    --train_manifest "$MANIFEST_DIR/stereo_train_boxfam.csv" \
    --val_manifest   "$MANIFEST_DIR/stereo_val_boxfam.csv" \
    --path_remap     "full_trunk:${DATASET}" \
    --H              280 \
    --W              512 \
    --base           64 \
    --pose_ch        32 \
    --epochs         80 \
    --batch_size     2 \
    --lr             3e-5 \
    --weight_decay   1e-4 \
    --patience       10 \
    --save_every     0 \
    --num_workers    4 \
    --grad_clip      1.0 \
    --min_depth      0.5 \
    --max_depth      10.0 \
    --lambda_mv      0.0 \
    --mv_warmup_epochs 2 \
    --seed           $SEED \
    --lr_warmup_epochs 5 \
    --out_dir        "$CKPT_ROOT/exp1" \
    --wandb \
    --wandb_project  "spur-adaptres-ablation" \
    --wandb_group    "${VARIANT}_${DSTAG}" \
    --wandb_run_name "${VARIANT}_${DSTAG}_seed${SEED}"

echo ""
echo "===== DONE  variant=${VARIANT}  seed=${SEED} ====="
date
