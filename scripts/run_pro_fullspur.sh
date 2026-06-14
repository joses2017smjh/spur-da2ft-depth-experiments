#!/bin/bash
#SBATCH --job-name=pro_fullspur
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-164,180
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.err

# Run PRO depth inference for the full_spur dataset (both barks), including
# RGB + Optical_flow stereo (_l/_r) pairs.
# Outputs land in full_spur/pro_refine/<bark>/<tree>/...
#
# Task mapping (NUM_TREES=100):
#   TASK_IDs   0-99:  bark_brown_02, trees 0-99
#   TASK_IDs 100-164: bark_willow_02, trees 0-64
#   TASK_ID  180:     bark_willow_02, tree 80
#   (gaps 165-179 are skipped cleanly via the missing-dir check)
#
# Submit: sbatch run_pro_fullspur.sh

set -euo pipefail

# --- Conda bootstrap ---
CANDIDATES=(
  "/nfs/stak/users/$USER/miniforge3"
  "/nfs/stak/users/$USER/miniforge"
  "$HOME/miniforge3"
  "$HOME/miniforge"
  "/nfs/stak/users/$USER/hpc-share/miniforge3"
)

CONDA_BASE=""
for p in "${CANDIDATES[@]}"; do
  if [ -f "$p/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="$p"
    break
  fi
done

if [ -z "$CONDA_BASE" ]; then
  echo "ERROR: conda.sh not found." >&2
  exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate depth-env

# Hardcode python path to avoid srun PATH-stripping issue
PYTHON="$CONDA_BASE/envs/depth-env/bin/python"
if [ ! -x "$PYTHON" ]; then
  echo "ERROR: python not found at $PYTHON" >&2
  exit 1
fi

# --- Caches / tmp ---
export TMPDIR=/nfs/hpc/share/sanchej7/tmp
mkdir -p "$TMPDIR"

export HF_HOME=/nfs/hpc/share/sanchej7/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

# --- Threading ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --- Paths ---
CV_ROOT="/nfs/hpc/share/sanchej7/Computer_Vision"
STAK_ROOT="/nfs/stak/users/sanchej7/hpc-share/Computer_Vision"

PRO_ROOT="${STAK_ROOT}/One-Look-is-Enough"
INPUT_BASE="${STAK_ROOT}/Data/full_spur/rgb"
OPTICAL_FLOW_BASE="${STAK_ROOT}/Data/full_spur/Optical_flow"
OUTPUT_ROOT="${STAK_ROOT}/Data/full_spur"

PY_SCRIPT="${CV_ROOT}/process_pro.py"

export PYTHONPATH="${PRO_ROOT}:${PRO_ROOT}/external:${CV_ROOT}:${PYTHONPATH:-}"

# --- Map array task -> (bark, tree) ---
# TASK_IDs 0-99   → bark_brown_02, trees 0-99
# TASK_IDs 100+   → bark_willow_02, trees (TASK_ID - 100)
BARKS=( "bark_brown_02" "bark_willow_02" )
NUM_TREES=100

TASK_ID="${SLURM_ARRAY_TASK_ID}"
BARK_IDX=$(( TASK_ID / NUM_TREES ))
TREE_IDX=$(( TASK_ID % NUM_TREES ))

BARK="${BARKS[$BARK_IDX]}"
TREE=$(printf "lpy_envy_%05d" "$TREE_IDX")

IN_DIR="${INPUT_BASE}/${BARK}/${TREE}"

echo "===== JOB START ====="
date
hostname
echo "TASK_ID=$TASK_ID  BARK=$BARK  TREE=$TREE"
echo "IN_DIR=$IN_DIR"
echo "python=$PYTHON"
"$PYTHON" -c "import torch; print('torch:', torch.__version__)" || true
nvidia-smi || true

if [ ! -d "$IN_DIR" ]; then
  echo "[SKIP] Missing: $IN_DIR"
  exit 0
fi

"$PYTHON" "$PY_SCRIPT" \
    --input_dir "$IN_DIR" \
    --output_root "$OUTPUT_ROOT" \
    --optical_flow_root "$OPTICAL_FLOW_BASE" \
    --pro_root "$PRO_ROOT" \
    --ckp_path "${PRO_ROOT}/pretrained/PRO/PRO.pth" \
    --patch_split_num 4 4 \
    --cai_mode m1 \
    --process_num 4 \
    --cam_subdirs \
    --overwrite

echo "===== JOB END ====="
date
