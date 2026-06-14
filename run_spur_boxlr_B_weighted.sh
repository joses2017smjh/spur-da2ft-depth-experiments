#!/bin/bash
#SBATCH --job-name=spur_boxlr_B_weighted
#SBATCH --array=1-5
#SBATCH --partition=dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Computer_Vision/logs/%x-%A_%a.err

# SPUR port of the bark02 boxlr A/B/C/D ablation.
# Same single-image DA2 fine-tune used for the trunk study, but on the SPUR
# bark_brown_02 box family (set_ids: box, box_cam1..8 = 9 set_ids), 10 trees
# x 9 set_ids x 6 shots = 540 samples / seed (80/20 tree split -> 432 train,
# 108 val). View: 50/50 seeded random _l/_r per (tree, set_id, shot) -- RGB
# pulled from Optical_flow/, depth/mask from their _l/_r files. No center.
# Box mask = global cam1 left-right union plate (designed for stereo views,
# so well-aligned with _l/_r sampling).
# Experiment B - weighted: per-pixel weights rebalance trunk vs box-only by count
# (n_trunk/n_box for box-only pixels, 1 for trunk).

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
EXP=B_weighted
export MANIFEST_DIR="/nfs/hpc/share/sanchej7/Computer_Vision/manifests/spur_boxlr_${EXP}_seed${SEED}"
CKPT_DIR="/nfs/hpc/share/sanchej7/Computer_Vision/checkpoints/spur_boxlr_${EXP}_seed${SEED}"
PRETRAINED="/nfs/hpc/share/sanchej7/Computer_Vision/da2_weights/depth_anything_v2_vitl.pth"
BOX_MASK_GLOBAL="/nfs/hpc/share/sanchej7/Computer_Vision/Data/box_mask_global/box_cam1_lr_union.png"
mkdir -p "$MANIFEST_DIR" "$CKPT_DIR" /nfs/hpc/share/sanchej7/Computer_Vision/logs

echo "===== ${EXP}  seed=${SEED} ====="; date; hostname; nvidia-smi; python --version

# Build manifests: SPUR bark_brown_02, full box family (box + box_cam1..8), 10 trees, 80/20 split.
if [ ! -f "$MANIFEST_DIR/spur_box_train.csv" ]; then
python3 - <<'PYEOF'
import os, csv, random
from pathlib import Path
seed = int(os.environ["SLURM_ARRAY_TASK_ID"])
mdir = Path(os.environ["MANIFEST_DIR"])
data_root = Path("/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur")
random.seed(seed)
N_TREES   = 10
SET_IDS   = ["box"] + [f"box_cam{i}" for i in range(1, 9)]   # 9 set_ids
SHOTS     = [f"shot{i:02d}" for i in range(1, 7)]
# Fixed 10-tree pool (sorted, first 10) -> same trees across all seeds, like the
# trunk study. Per-seed randomness only affects the 80/20 split inside that pool.
all_trees = sorted(p.name for p in (data_root / "rgb" / "bark_brown_02").iterdir())
pool = all_trees[:N_TREES]
print(f"Tree pool ({len(pool)} trees): {pool}")
rows = []
for tree in pool:
    for sid in SET_IDS:
        for shot in SHOTS:
            # Random per (tree, set_id, shot) -- 50/50 _l vs _r, seeded.
            # RGB comes from Optical_flow/ (stereo l/r views), depth and mask
            # from the matching _l or _r files in their respective dirs.
            lr  = random.choice(["l", "r"])
            rgb = data_root / "Optical_flow" / "bark_brown_02" / tree / sid / f"{tree}_{shot}_{lr}.png"
            dep = data_root / "depth"        / "bark_brown_02" / tree / sid / f"{tree}_{shot}_{lr}.npy"
            mk  = data_root / "mask"         / "bark_brown_02" / tree / sid / f"{tree}_{shot}_{lr}.png"
            if not (rgb.exists() and dep.exists() and mk.exists()): continue
            rows.append(dict(bark="bark_brown_02", tree=tree, set_id=sid, shot=shot, lr=lr,
                             rgb_path=str(rgb), depth_path=str(dep), mask_path=str(mk)))
shuffled = pool[:]; random.shuffle(shuffled)
n_train = max(1, int(0.8 * len(shuffled)))                    # 8/2 split
tr, va = set(shuffled[:n_train]), set(shuffled[n_train:])
print(f"Seed {seed}: train trees={sorted(tr)}  val trees={sorted(va)}")
from collections import Counter
print(f"l/r split: {dict(Counter(r['lr'] for r in rows))}")
fields = ["bark","tree","set_id","shot","lr","rgb_path","depth_path","mask_path"]
for name, ts in [("train", tr), ("val", va)]:
    sel = sorted([r for r in rows if r["tree"] in ts], key=lambda r:(r["tree"], r["set_id"], r["shot"]))
    with open(mdir / f"spur_box_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(sel)
    print(f"  {name}: {len(sel):,} rows")
PYEOF
fi

python train_depth_da2.py \
    --encoder           vitl \
    --pretrained-from   "$PRETRAINED" \
    --train-manifest    "$MANIFEST_DIR/spur_box_train.csv" \
    --val-manifest      "$MANIFEST_DIR/spur_box_val.csv" \
    --save-path         "$CKPT_DIR" \
    --epochs            80 \
    --bs                2 \
    --lr                0.000005 \
    --num-workers       4 \
    --seed              $SEED \
    --train-box-mask \
    --box-mask-global   "$BOX_MASK_GLOBAL" \
    --box-loss-mode     weighted \
    --eval-box-mask \
    --wandb-project     "spur-boxlr-ablation" \
    --wandb-run-name    "spur_boxlr_${EXP}_seed${SEED}"

echo "===== DONE  ${EXP}  seed=${SEED} ====="; date
