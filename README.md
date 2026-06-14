# spur-da2ft-depth-experiments

Vision-based **metric depth estimation for robotic apple-tree pruning**.
This repository contains the synthetic-data pipeline, Depth Anything V2
fine-tuning code, and stereo / multi-view refinement ablations
(CNN U-Net and DINOv2 + depth-side branch) described in the accompanying
CVPR-style report.

### Built With

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Blender](https://img.shields.io/badge/Blender-4.2.13-F5792A?style=for-the-badge&logo=blender&logoColor=white)](https://www.blender.org/)
[![DINOv2](https://img.shields.io/badge/DINOv2-ViT--L-5D3FD3?style=for-the-badge)](https://github.com/facebookresearch/dinov2)
[![Depth%20Anything%20V2](https://img.shields.io/badge/Depth%20Anything-V2-009688?style=for-the-badge)](https://github.com/DepthAnything/Depth-Anything-V2)

## 📑 Table of Contents

- [Project Identity](#-project-identity)
- [Value Proposition](#-value-proposition)
- [Core Features and Benefits](#core-features-and-benefits)
- [Access and Usage](#-access-and-usage)
- [Generating the Dataset](#-generating-the-dataset)
- [Running the Experiments](#-running-the-experiments)
- [Hardware Constraints](#-hardware-constraints)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Scripts](#scripts)
- [Development Challenges and Solutions](#development-challenges-and-solutions)
- [Developer References](#developer-references)
- [Contact](#contact)

---

## 📌 Project Identity

### Active Team & Roles
- **Jose Sanchez** — Sole contributor (Oregon State University)

### Timeline / Status
- **Jan – Mar 2026**: Synthetic data pipeline + zero-shot baselines ✅
- **Mar – May 2026**: DA2 fine-tune + box-anchor loss study ✅
- **May – Jun 2026**: Stereo / multi-view refinement (CNN + DINOv2) ✅
- **Jun 2026**: Report + reproducible release 📝 *(current)*

## 🌟 Value Proposition

### Problem Statement
Robotic pruning of dormant apple trees requires per-pixel metric depth on
**thin branches against cluttered backgrounds**, a regime where real depth
sensors return sparse, noisy data and off-the-shelf monocular models drift
in scale. This project asks: *given limited real-world data, where does the
biggest accuracy gain come from — post-hoc calibration, targeted synthetic
fine-tuning, or stereo / multi-view refinement?*

### Target Audience

| Audience       | Needs                                                   |
| ---            | ---                                                     |
| Robotics teams | Metric depth on tree structure for arm/end-effector planning |
| Researchers    | Reproducible ablation of calibration vs. fine-tune vs. refinement |
| Agri-vision    | Synthetic orchard data pipeline with full pose + GT depth |
| Students       | Working end-to-end depth pipeline (Blender → train → eval) |

### Core Features and Benefits

#### 1. Synthetic Orchard Generator (Blender 4.2.13)
A scripted pipeline produces RGB, ground-truth depth, segmentation masks,
camera intrinsics, and poses for 100 Envy/UFO tree models with four bark
textures and configurable camera rigs (trunk-only, box-anchor, stereo, multi-view).

#### 2. Calibration + Preprocessing Sweep
Global linear α/β alignment of DA3 Metric / DA3 Relative / PatchRefineOnce
(PRO) outputs with a winsor × erosion × min-std × fit-space sweep, plus a
**box-anchor** strategy that uses a fixed known object as a metric reference.

#### 3. Depth Anything V2 Fine-Tune
Trunk-masked SiLog fine-tune (single image) with four box-loss variants
(union / weighted / balanced / anchor) that include the box pixels at
different weights.

#### 4. Stereo & Multi-View Refinement
- **Shared CNN stereo U-Net** (`MVStereoUNet`) — baseline RGB+D refiner.
- **DINOv2 (frozen) + Depth-Side Branch** (`MVStereoDINOUNet`) — pretrained
  RGB encoder paired with a trainable depth pathway and learned 1×1
  cross-view fusion at the bottleneck.

#### 5. Reproducible Ablation Harness
A consistent 10-tree / 80-20 split / 5-seed / 80-epoch / patience-10 protocol
runs on every encoder × input × depth source × fusion combination via
`run_spur_*.sh` scripts.

## 📘 Access and Usage

### Prerequisites
1. **Python 3.10** (we ship a `requirements.txt`).
2. **CUDA GPU with ≥24 GB VRAM** for ViT-L training (see [hardware](#-hardware-constraints)).
3. **Blender 4.2.13** for synthetic data generation (only needed if regenerating).
4. **Conda / Miniforge** for the Python environment.

### Environment Setup
```bash
git clone https://github.com/joses2017smjh/spur-da2ft-depth-experiments.git
cd spur-da2ft-depth-experiments

conda create -n spur python=3.10
conda activate spur
pip install -r requirements.txt
```

Place the **Depth Anything V2 ViT-L** pretrained weights at:
```
da2_weights/depth_anything_v2_vitl.pth
```
(Download from the official DA2 release.)

### Configuration
Top-level paths and flags are exposed in each `run_spur_*.sh`:
```bash
# Dataset selection
export DATASET=full_trunk          # or full_spur
# Input depth source for refiners
export INPUT_DEPTH_SUBDIR=Da2Finetune   # or pro_refine
```

## 🌳 Generating the Dataset

### Required Assets
The Blender pipeline needs three things on disk:

| Asset                  | Where                              | Source |
| ---                    | ---                                | --- |
| Blender scene          | `orchard_template.blend`           | shipped in repo |
| Per-tree geometry      | `trees/ply/lpy_envy_*.ply`         | provided in repo |
| Per-tree metadata JSON | `trees/metadata/lpy_envy_*_metadata.json` | cylinder info (radius, centroid, trunk/branch/spur tag) |
| Bark textures          | `textures/<bark_name>/<bark_name>_diff_4k.jpg` and `_nor_gl_4k.exr` | 4 textures: `bark_brown`, `bark_brown_02`, `bark_willow`, `bark_willow_02` |

### Key Flags in `Dataloader/generate_tree2.py`

| Knob | Default | What it does |
| --- | --- | --- |
| `BARK_NAME`             | env | Which bark texture set to apply (`bark_brown_02`, etc.). |
| `TREE_ID`               | env | Which `lpy_envy_*` to render. |
| `CV_OUTPUT_DIR`         | env | Output root; produces `rgb/`, `depth/`, `mask/`, `Optical_flow/`, `ann/`, `box_mask/`. |
| `CV_FORCE_RENDER=1`     | env | Re-render even if outputs already exist. |
| `process_res`           | 504 | Internal depth processing resolution. |
| `TEXTURES_DIR`          | const | Directory of bark textures. |
| `RENDER_BOX = True`     | const | Adds the fixed box anchor (~30 cm in front of camera, 15×7 cm). |
| `SHOTS`                 | const | Six fixed camera poses per (tree, set_id). |
| `SET_IDS`               | const | `box`, `box_cam1..8`, `cam1..10`. |

### Workflow
```bash
# 1. Render one tree × one texture (smoke test)
BARK_NAME=bark_brown_02 \
TREE_ID=lpy_envy_00001 \
CV_OUTPUT_DIR=./Data/full_spur \
CV_FORCE_RENDER=1 \
blender -b orchard_template.blend -P Dataloader/generate_tree2.py

# 2. Render all 100 trees in a texture
for i in $(seq -f "%05g" 0 99); do
  BARK_NAME=bark_brown_02 TREE_ID=lpy_envy_$i \
  CV_OUTPUT_DIR=./Data/full_spur \
  blender -b orchard_template.blend -P Dataloader/generate_tree2.py
done

# 3. Generate PRO depth maps (input for refiners)
bash run_pro_fullspur.sh
```

Output layout:
```
Data/<dataset>/
  rgb/<bark>/<tree>/<set_id>/<tree>_<shot>.png
  Optical_flow/<bark>/<tree>/<set_id>/<tree>_<shot>_{l,r}.png
  depth/<bark>/<tree>/<set_id>/<tree>_<shot>_{l,r,_}.npy
  mask/<bark>/<tree>/<set_id>/<tree>_<shot>_{l,r,_}.png
  ann/<bark>/<tree>/<set_id>/<tree>_<shot>_{l,r}.json
  pro_refine/...      (after PRO inference)
  Da2Finetune/...     (after DA2 inference)
```

## ▶️ Running the Experiments

Each experiment is a `run_spur_*.sh` script that builds a per-seed manifest
and launches `train_depth_da2.py` (single-image) or
`MVP_MODEL/train_mvp_stereo{,_dino}.py` (refiner). Shared protocol: 10 trees,
8/2 split, 5 seeds (array tasks), 80 epochs, patience 10, no pose,
nearest-GT loader.

| Experiment family | Script |
| --- | --- |
| DA2 box-anchor loss (A/B/C/D) | `run_spur_boxlr_{A_union,B_weighted,C_balanced,D_anchor}.sh` |
| DA2 view-diversity | `run_spur_da2_{halfhalf,boxfam,camonly}_nomv_seeds.sh` |
| DINO RGB / RGB+D refiner ablations | `run_spur_dino_*_seeds.sh` |
| CNN refiner ablations | `run_spur_cnn_*_seeds.sh` |
| Multi-pair sweep | `run_spur_dino_da2ft_{2,3,4}pair_fusion_nopose.sh` |

Local (no SLURM):
```bash
SLURM_ARRAY_TASK_ID=1 bash run_spur_dino_da2ft_1pair_fusion_nopose_dgx2.sh
```

## 🖥️ Hardware Constraints

| Stage | Minimum | Recommended |
| --- | --- | --- |
| **Blender render** (per tree) | 16 GB RAM, any CUDA GPU (8 GB) | 32 GB RAM, RTX 3060+ |
| **PRO / DA2 inference** | 12 GB VRAM | 24 GB+ VRAM |
| **DA2 fine-tune (ViT-L, bs=2)** | **24 GB VRAM** | A40 / A100 / H100 / V100-32GB |
| **CNN stereo refiner (1 pair)** | 12 GB VRAM | 24 GB |
| **DINO refiner (1 pair, frozen ViT-L)** | 16 GB VRAM | 24 GB+ |
| **DINO refiner (3–4 pair)** | 24 GB VRAM | 40 GB+ (A100 / H100) |
| **Disk for full dataset** | — | ~250 GB |
| **CPU / RAM during training** | 8 cores, 32 GB | 16 cores, 64 GB |

Floor for a single-GPU personal computer: an **RTX 3090/4090 (24 GB)** can
run every experiment in this repo at `bs=2, H=280, W=512`, just slower than
the cluster (~2–4× wall-time vs. A40). Anything with **<16 GB VRAM** will
need reduced batch size, smaller image size, or to skip DA2 fine-tune and
use a released checkpoint.

## Architecture

```
+---------------------+      +-----------------------+      +-------------------+
|  Blender Renderer   | ---> |   Synthetic Dataset   | ---> | PRO / DA2 Predictor|
|  generate_tree2.py  |      |  rgb / depth / mask / |      |   (preprocess)    |
+---------------------+      |  Optical_flow / ann   |      +-------------------+
                             +-----------------------+                |
                                       |                              v
                                       v                  +-------------------+
                             +-----------------------+    |  Calibration Sweep|
                             |  Single-Image FT (DA2)|    |  (α/β + preproc)  |
                             +-----------------------+    +-------------------+
                                       |
                                       v
                             +-----------------------+      +-------------------+
                             |  Stereo / Multi-View  |<---->|  Loader override  |
                             |  Refiner (CNN / DINO) |      |  INPUT_DEPTH_SUBDIR|
                             +-----------------------+      +-------------------+
```

## Repository Layout

```
spur-da2ft-depth-experiments/
├── README.md
├── requirements.txt
├── orchard_template.blend          # Blender scene
├── Dataloader/
│   ├── generate_tree2.py
│   ├── move_camera.py
│   └── exr_to_npy.py
├── trees/
│   ├── ply/        lpy_envy_*.ply
│   └── metadata/   lpy_envy_*_metadata.json
├── textures/
│   ├── bark_brown_02/
│   ├── bark_willow_02/
│   ├── bark_brown/
│   └── bark_willow/
├── da2_weights/                    # user provides depth_anything_v2_vitl.pth
├── depth-anything-v2/              # vendored DA2 metric_depth code
├── dataset/                        # PyTorch datasets
│   ├── trunk_da2.py
│   ├── trunk_stereo_mvp.py
│   ├── trunk_stereo_pair_mvp.py
│   ├── trunk_stereo_triplet_mvp.py
│   └── trunk_stereo_quad_mvp.py
├── MVP_MODEL/
│   ├── mvp_stereo_model.py         # MVStereoUNet (CNN)
│   ├── mvp_stereo_dino_model.py    # MVStereoDINOUNet (frozen DINOv2 + DSB)
│   ├── train_mvp_stereo.py
│   └── train_mvp_stereo_dino.py
├── train_depth_da2.py              # DA2 single-image fine-tune
├── run_spur_*.sh                   # all experiment launchers
└── manifests/source/               # canonical 80/20 stereo manifests
    ├── stereo_train_manifest.csv
    └── stereo_val_manifest.csv
```

## Scripts

| Path | Purpose |
| --- | --- |
| `run_spur_boxlr_*.sh`              | DA2 box-anchor loss (A/B/C/D) ablation |
| `run_spur_da2_*.sh`                | DA2 view-diversity (data composition) |
| `run_spur_cnn_*.sh`                | CNN refiner ablations |
| `run_spur_dino_*.sh`               | DINO refiner ablations |

## Development Challenges and Solutions

**Dataset reorganization drift.** Cached manifests referenced trees that were
later moved/deleted. **Solution:** scripts guard manifest reuse with an
existence check; running with deleted-tree state requires
`rm -rf manifests/<variant>_seed*` to force rebuild.

**Bilinear GT-resize bleed.** The stereo loader's `_load_depth` originally
used `mode="bilinear"`, which averaged 2 m trunk pixels with the (zeroed)
background sentinel at silhouettes — corrupting ~5 % of trunk-mask edge
pixels and inflating masked RMSE ~20×. **Solution:** switched GT resize to
`nearest`; verified 0 % >10 m pixels in masked region.

**Pose ↔ fusion coupling.** `MVStereoDINOUNet` ties
`use_pose = use_pose and not no_fusion` because pose is injected at the
cross-view bottleneck. **Solution:** documented the coupling; pose-OFF runs
were re-launched with both `--no_pose` and explicit fusion choice.

**Multi-pair dataset for CNN.** The CNN trainer originally required indexed
columns for multi-pair. **Solution:** extended `train_mvp_stereo.py` to use
the nearest-neighbour Pair/Triplet/Quad datasets the DINO trainer already
uses.

## Developer References

- **Depth Anything V2** — Yang et al., 2024.
  <https://github.com/DepthAnything/Depth-Anything-V2>
- **Depth Anything V3** — Lin et al., 2025.
  <https://github.com/depth-anything/Depth-Anything-V3>
- **PatchRefineOnce (PRO)** — Kwon et al., 2025.
  <https://github.com/inkyu-kwon/PatchRefineOnce>
- **DINOv2** — Oquab et al., 2023.
  <https://github.com/facebookresearch/dinov2>
- **U-Net** — Ronneberger et al., 2015.
  <https://arxiv.org/abs/1505.04597>
- **TilingZoeDepth** — Bill F. Smith.
  <https://github.com/BillFSmith/TilingZoeDepth>

## Contact

Jose Sanchez — sanchej7@oregonstate.edu
Oregon State University
