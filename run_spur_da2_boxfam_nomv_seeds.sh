#!/bin/bash
#SBATCH --job-name=spur_da2_boxfam_v4_10trees
#SBATCH --array=1-5
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.err

# v4: DA2 fine-tune on full_spur box family (box + box_cam1..8 = 9 set_ids),
# 10 trees, 80/20 tree-level split, 5 seeds. Single-image training via
# train_depth_da2.py (matches the boxlr ablation protocol). Per (tree, set_id,
# shot) the manifest randomly picks one of the stereo views {_l, _r} with 50/50
# probability (seeded), giving 10 * 9 * 6 = 540 samples / seed (432 train,
# 108 val with 8/2 split). No box-plate supervision -- this is view-diversity.

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
export MANIFEST_DIR="/nfs/hpc/share/sanchej7/Computer_Vision/manifests/spur_boxfam_v4_seed${SEED}"
CKPT_DIR="/nfs/hpc/share/sanchej7/Computer_Vision/checkpoints/spur_da2_boxfam_v4_seed${SEED}"
PRETRAINED="/nfs/hpc/share/sanchej7/Computer_Vision/da2_weights/depth_anything_v2_vitl.pth"
export SPUR_ROOT="/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur"
mkdir -p "$MANIFEST_DIR" "$CKPT_DIR" /nfs/hpc/share/sanchej7/Computer_Vision/logs

echo "===== JOB START  seed=${SEED} ====="; date; hostname; nvidia-smi; python --version

if [ ! -f "$MANIFEST_DIR/spur_boxfam_train.csv" ]; then
python3 - <<'PYEOF'
import os, csv, random
from pathlib import Path
seed      = int(os.environ["SLURM_ARRAY_TASK_ID"])
mdir      = Path(os.environ["MANIFEST_DIR"])
spur_root = Path(os.environ["SPUR_ROOT"])
random.seed(seed)

N_TREES    = 10
BOX_FAMILY = ["box"] + [f"box_cam{i}" for i in range(1, 9)]   # 9 set_ids
BARK       = "bark_brown_02"
SHOTS      = [f"shot{i:02d}" for i in range(1, 7)]

of_root   = spur_root / "Optical_flow" / BARK
dep_root  = spur_root / "depth"        / BARK
mask_root = spur_root / "mask"         / BARK

# Fixed 10-tree pool (sorted, first 10) -> same trees across all seeds.
all_trees = sorted(p.name for p in of_root.iterdir() if p.is_dir())
pool = all_trees[:N_TREES]
print(f"Tree pool ({len(pool)}): {pool}")

rows = []
for tree in pool:
    for sid in BOX_FAMILY:
        of_dir   = of_root   / tree / sid
        dep_dir  = dep_root  / tree / sid
        mask_dir = mask_root / tree / sid
        if not of_dir.exists():
            continue
        for shot in SHOTS:
            # Random per (tree, set_id, shot) -- 50/50 _l vs _r, seeded.
            lr = random.choice(["l", "r"])
            rgb  = of_dir   / f"{tree}_{shot}_{lr}.png"
            dep  = dep_dir  / f"{tree}_{shot}_{lr}.npy"
            mk   = mask_dir / f"{tree}_{shot}_{lr}.png"
            if not (rgb.exists() and dep.exists() and mk.exists()):
                continue
            rows.append({
                "bark": BARK, "tree": tree, "set_id": sid, "shot": shot, "lr": lr,
                "rgb_path": str(rgb), "depth_path": str(dep), "mask_path": str(mk),
            })

shuffled = pool[:]; random.shuffle(shuffled)
n_train  = max(1, int(0.8 * len(shuffled)))
tr_set, va_set = set(shuffled[:n_train]), set(shuffled[n_train:])
print(f"Seed {seed}: train trees={sorted(tr_set)}  val trees={sorted(va_set)}")
from collections import Counter
lr_tr = Counter(r["lr"] for r in rows if r["tree"] in tr_set)
lr_va = Counter(r["lr"] for r in rows if r["tree"] in va_set)
print(f"Train l/r split: {dict(lr_tr)}   Val l/r split: {dict(lr_va)}")

fields = ["bark", "tree", "set_id", "shot", "lr",
          "rgb_path", "depth_path", "mask_path"]
for name, ts in [("train", tr_set), ("val", va_set)]:
    sel = sorted([r for r in rows if r["tree"] in ts],
                 key=lambda r: (r["tree"], r["set_id"], r["shot"]))
    with open(mdir / f"spur_boxfam_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(sel)
    print(f"  {name}: {len(sel):,} rows")
PYEOF
fi

echo "Manifests ready (seed=${SEED})."

python train_depth_da2.py \
    --encoder           vitl \
    --pretrained-from   "$PRETRAINED" \
    --train-manifest    "$MANIFEST_DIR/spur_boxfam_train.csv" \
    --val-manifest      "$MANIFEST_DIR/spur_boxfam_val.csv" \
    --save-path         "$CKPT_DIR" \
    --epochs            80 \
    --bs                2 \
    --lr                0.000005 \
    --num-workers       4 \
    --seed              $SEED \
    --wandb-project     "bark02-full-spur-ablation" \
    --wandb-run-name    "spur_da2_boxfam_v4_seed${SEED}"

echo "===== DONE  seed=${SEED} ====="; date
