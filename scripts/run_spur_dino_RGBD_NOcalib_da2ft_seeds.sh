#!/bin/bash
#SBATCH --job-name=spur_dino_RGBD_NOcalib_da2ft
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# ============================================================================
# SPUR DINOv2 RGB+D with DA2-FINETUNED input depth, full_spur box-family.
#   Model:  MVStereoDINOUNet (DINOv2 ViT-L frozen + DepthSideBranch + DINODecoder)
#   Fusion: ON (default cross-view bottleneck fusion).
#   Input depth: DA2-finetuned predictions (Da2Finetune/), NOT PRO.
#   Calibration: --no_pro_calib (no alpha/beta; DA2 depth is already metric).
#   5 seeds, 10 trees/seed (8 train + 2 val), 80/20, 80 epochs, patience 10.
#   NOTE: DINOv2 ViT-L weights are fetched at runtime via torch.hub
#   (facebookresearch/dinov2) -> needs internet on first run (cached in TORCH_HOME).
#
# ---------------------------------------------------------------------------
# HOW TO RUN
#   1. Activate a Python 3.10 env with the deps (see README.md). Python 3.10 is
#      REQUIRED: the model classes ship as 3.10 bytecode (_mvp_precompiled.pyc).
#   2. Untar the dataset somewhere and point DATA_ROOT at the dir that
#      CONTAINS full_spur/  (i.e. DATA_ROOT/full_spur/depth/..., /Da2Finetune/...)
#   3a. SLURM array (all 5 seeds):
#         DATA_ROOT=/path/to/data sbatch run_spur_dino_RGBD_NOcalib_da2ft_seeds.sh
#   3b. Plain bash (one seed, no SLURM):
#         DATA_ROOT=/path/to/data SEED=1 bash run_spur_dino_RGBD_NOcalib_da2ft_seeds.sh
#
# ENV VARS
#   DATA_ROOT  (required) dir containing full_spur/
#   REPO_ROOT  (auto)     repo location; holds code + manifests/source/
#   OUT_ROOT   (default REPO_ROOT/outputs) where manifests + checkpoints go
#   SEED       (default 1) used only when not running under SLURM
#   WANDB_API_KEY (optional) export to enable W&B logging; unset => W&B off
# ============================================================================

set -euo pipefail

# ── Roots ───────────────────────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
: "${DATA_ROOT:?Set DATA_ROOT to the dir containing full_spur/ (where you untarred the data)}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/outputs}"
cd "$REPO_ROOT"

# Original absolute Data prefix baked into manifests/source/*.csv. The loader
# rewrites this prefix (and full_trunk -> full_spur) to your DATA_ROOT.
SRC_DATA_PREFIX="/nfs/stak/users/sanchej7/hpc-share/Computer_Vision/Data"

# ── Python env ───────────────────────────────────────────────────────────────
# Activate your own Python 3.10 environment before running, e.g.:
#   conda activate depth-env
# (left to the user so this script is machine-agnostic).

export TMPDIR="${TMPDIR:-/tmp}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

NTHREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS=$NTHREADS OPENBLAS_NUM_THREADS=$NTHREADS MKL_NUM_THREADS=$NTHREADS

# ── Seed (SLURM array task id, or SEED when run as plain bash) ────────────────
SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-1}}"

DATASET="${DATASET:-full_spur}"
DSTAG="${DATASET#full_}"
export DATASET

VARIANT="dino_RGBD_NOcalib_da2ft"

# Redirect the stereo loader's input-depth source from pro_refine/ to the
# DA2-finetuned depth maps.
export INPUT_DEPTH_SUBDIR="Da2Finetune"

export SRC_STEREO_TRAIN="$REPO_ROOT/manifests/source/stereo_train_manifest.csv"
export SRC_STEREO_VAL="$REPO_ROOT/manifests/source/stereo_val_manifest.csv"
export MANIFEST_DIR="$OUT_ROOT/manifests/${VARIANT}_${DSTAG}_seed${SEED}"
export CKPT_ROOT="$OUT_ROOT/checkpoints/${VARIANT}_${DSTAG}_seed${SEED}"
export SRC_DATA_PREFIX DATA_ROOT SEED

mkdir -p "$MANIFEST_DIR" "$CKPT_ROOT/exp1" "$REPO_ROOT/logs"

echo "===== JOB START  variant=${VARIANT}  dataset=${DATASET}  seed=${SEED} ====="
date; hostname; nvidia-smi || true; python --version

# ─────────────────────────────────────────────────────────────────────────────
# Build box-family 2-view stereo manifests for this seed
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [seed=${SEED}] Building box-family 2-view stereo manifests ==="

python3 - <<'PYEOF'
import csv, random, os
import cv2

seed = int(os.environ["SEED"])
random.seed(seed)

SRC_TRAIN    = os.environ["SRC_STEREO_TRAIN"]
SRC_VAL      = os.environ["SRC_STEREO_VAL"]
MANIFEST_DIR = os.environ["MANIFEST_DIR"]
DATASET      = os.environ.get("DATASET", "full_spur")
SRC_PREFIX   = os.environ["SRC_DATA_PREFIX"]
DATA_ROOT    = os.environ["DATA_ROOT"]

# Manifests reference <SRC_PREFIX>/full_trunk/...; relocate to
# <DATA_ROOT>/full_spur/... so the on-disk mask_ok() check finds the files.
OLD = f"{SRC_PREFIX}/full_trunk"
NEW = f"{DATA_ROOT}/{DATASET}"

PATH_KEYS = ("rgb_path", "depth_path", "mask_path", "ann_path",
             "pair_rgb_path", "pair_depth_path", "pair_mask_path", "pair_ann_path")

def remap(row):
    for k in PATH_KEYS:
        if k in row and row[k]:
            row[k] = row[k].replace(OLD, NEW)
    return row

BOX_FAMILY = {
    "box",
    "box_cam1", "box_cam2", "box_cam3", "box_cam4",
    "box_cam5", "box_cam6", "box_cam7", "box_cam8",
}

def read_csv(path):
    with open(path, newline="") as fh:
        return [remap(r) for r in csv.DictReader(fh)]

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
# Optional W&B logging — enabled only if WANDB_API_KEY is exported.
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_DIR="${WANDB_DIR:-$TMPDIR}"
    WANDB_ARGS=(--wandb
        --wandb_project  "spur-da2ft-depth-ablation"
        --wandb_group    "${VARIANT}_${DSTAG}"
        --wandb_run_name "${VARIANT}_${DSTAG}_seed${SEED}")
else
    echo "WANDB_API_KEY not set -> W&B logging disabled."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Training — MVStereoDINOUNet (frozen DINOv2 ViT-L + DepthSideBranch), RGB+D
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [seed=${SEED}] Training ${VARIANT} ==="
python MVP_MODEL/train_mvp_stereo_dino.py \
    --n_views        2 \
    --no_pro_calib \
    --train_manifest "$MANIFEST_DIR/stereo_train_boxfam.csv" \
    --val_manifest   "$MANIFEST_DIR/stereo_val_boxfam.csv" \
    --path_remap     "${SRC_DATA_PREFIX}/full_trunk:${DATA_ROOT}/${DATASET}" \
    --H              280 \
    --W              512 \
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
    --seed           "$SEED" \
    --lr_warmup_epochs 5 \
    --out_dir        "$CKPT_ROOT/exp1" \
    "${WANDB_ARGS[@]}"

echo ""
echo "===== DONE  variant=${VARIANT}  seed=${SEED} ====="
date
