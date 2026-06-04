# SPUR DA2-finetuned depth ablation — CNN vs DINO (box family)

Two stereo trunk-depth refinement experiments on the synthetic SPUR orchard
dataset (`bark_brown_02`, box camera family), using **DA2-finetuned** input
depth. Each is a 5-seed sweep; per seed it samples 8 train + 2 val trees from an
80/20 tree-level split, builds 2-view stereo manifests, and trains for up to 80
epochs (patience 10) with SiLog loss on the trunk-masked GT depth.

| Script | Model | Trainer |
|---|---|---|
| `run_spur_cnn_fusion_RGBD_NOcalib_da2ft_seeds.sh` | `MVStereoUNet` — from-scratch 4-ch CNN (RGB+depth), cross-view fusion ON | `MVP_MODEL/train_mvp_stereo.py` |
| `run_spur_dino_RGBD_NOcalib_da2ft_seeds.sh` | `MVStereoDINOUNet` — frozen DINOv2 ViT-L + DepthSideBranch + DINODecoder, fusion ON | `MVP_MODEL/train_mvp_stereo_dino.py` |

Both use `--no_pro_calib` (DA2 depth is already metric, no α/β), input resolution
H=280×W=512, AdamW lr=3e-5 wd=1e-4, 5-epoch warmup, grad clip 1.0, batch 2.
This repo ships the **clean** GT-target interpolation (`nearest`, not `bilinear`),
so it reproduces the clean runs.

### Reference results (best val masked RMSE, mean ± sd over 5 seeds)
| Variant | Best val RMSE | DA2 input floor |
|---|---|---|
| CNN, clean GT | 0.057 ± 0.004 | 0.0556 m |
| DINO, clean GT | **0.048 ± 0.007** | 0.0556 m |

DINO beats the CNN (which sits on the DA2 input floor) by ~14%.

---

## Requirements

> **Python 3.10 is mandatory.** The core model classes (UNet/DINO encoders,
> decoders, loss functions) ship only as Python 3.10 bytecode in
> `MVP_MODEL/_mvp_precompiled.pyc` — the original source was lost and the
> included `_mvp_precompiled_decompiled.py` is a *partial* reconstruction
> (reference only). The `.pyc` will not import under 3.9 or 3.11+.

- A CUDA GPU. Validated on a single A40 (48 GB); ~64 GB host RAM, 8 CPUs.
- Internet access on first DINO run: the DINOv2 ViT-L backbone is fetched via
  `torch.hub` (`facebookresearch/dinov2`) and cached in `$TORCH_HOME`.

```bash
conda create -n spur python=3.10 -y && conda activate spur
# install a CUDA-matched torch first (see https://pytorch.org), then:
pip install -r requirements.txt
```

---

## Repo layout
```
run_spur_cnn_fusion_RGBD_NOcalib_da2ft_seeds.sh   # CNN experiment
run_spur_dino_RGBD_NOcalib_da2ft_seeds.sh         # DINO experiment
requirements.txt
MVP_MODEL/
  train_mvp_stereo.py            # CNN trainer
  train_mvp_stereo_dino.py       # DINO trainer
  mvp_stereo_model.py            # MVStereoUNet
  mvp_stereo_dino_model.py       # MVStereoDINOUNet
  mvp_depth_model_Unet.py        # stub -> loads classes from _mvp_precompiled.pyc
  mvp_depth_refine_model.py      # stub -> loads classes from _mvp_precompiled.pyc
  _mvp_precompiled.pyc           # REQUIRED Python 3.10 bytecode (model classes)
  _mvp_precompiled_decompiled.py # partial reconstruction, reference only
dataset/
  trunk_stereo_mvp.py            # 2-view stereo dataset (used by both)
  trunk_stereo_triplet_mvp.py    # imported by DINO trainer (3-view; unused at n_views=2)
  trunk_stereo_quad_mvp.py       # imported by DINO trainer (4-view; unused at n_views=2)
manifests/source/
  stereo_train_manifest.csv      # 80 trees (2400 box-family rows)
  stereo_val_manifest.csv        # 20 trees (600 box-family rows)
```

---

## Data (shipped separately — ~88 GB)

The image/depth data is far too large for git and is **not** in this repo — it's
copied separately via `rsync`. See **[TRANSFER.md](TRANSFER.md)** for the exact
commands (incl. a no-WSL Windows path) and the file list
[`data_transfer_filelist.txt`](data_transfer_filelist.txt). Per modality, the
box-family subset is:

| Modality | Path under `full_spur/` | Size | Role |
|---|---|---|---|
| GT depth | `depth/`        | 49.8 GB | training target (`.npy`) |
| Input depth | `Da2Finetune/` | 20.9 GB | model input depth (`.npy`) |
| RGB | `Optical_flow/`     | 15.9 GB | model input RGB (`.png`) |
| ann | `ann/`              | 0.8 GB  | camera annotations (`.json`) |
| mask | `mask/`            | 0.2 GB  | trunk masks (`.png`) |

After untarring you must end up with this layout:
```
<DATA_ROOT>/full_spur/depth/bark_brown_02/<tree>/<box*>/*.npy
<DATA_ROOT>/full_spur/Da2Finetune/bark_brown_02/<tree>/<box*>/*.npy
<DATA_ROOT>/full_spur/Optical_flow/bark_brown_02/<tree>/<box*>/*.png
<DATA_ROOT>/full_spur/ann/bark_brown_02/<tree>/<box*>/*.json
<DATA_ROOT>/full_spur/mask/bark_brown_02/<tree>/<box*>/*.png
```
The manifests store the original absolute paths; the run scripts rewrite that
prefix to your `$DATA_ROOT` automatically (`--path_remap` + the manifest-builder
remap), so you only set `DATA_ROOT` — no manifest editing needed.

---

## Running

Set `DATA_ROOT` to the directory that **contains** `full_spur/`.

**SLURM (all 5 seeds as an array):**
```bash
sbatch --export=ALL,DATA_ROOT=/path/to/data run_spur_cnn_fusion_RGBD_NOcalib_da2ft_seeds.sh
sbatch --export=ALL,DATA_ROOT=/path/to/data run_spur_dino_RGBD_NOcalib_da2ft_seeds.sh
```
(Add `--partition=<yours>` — no partition is hardcoded.)

**Plain bash (single seed, no SLURM):**
```bash
DATA_ROOT=/path/to/data SEED=1 bash run_spur_cnn_fusion_RGBD_NOcalib_da2ft_seeds.sh
DATA_ROOT=/path/to/data SEED=1 bash run_spur_dino_RGBD_NOcalib_da2ft_seeds.sh
```

### Environment variables
| Var | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | *(required)* | dir containing `full_spur/` |
| `REPO_ROOT` | auto (script dir) | repo location (code + `manifests/source/`) |
| `OUT_ROOT`  | `REPO_ROOT/outputs` | where per-seed manifests + checkpoints are written |
| `SEED`      | `1` | seed used only when not under SLURM |
| `WANDB_API_KEY` | unset | export to enable W&B logging; unset ⇒ logging off |

Outputs land in `OUT_ROOT/checkpoints/<variant>_spur_seed<N>/exp1/` (best
checkpoint + metrics); per-seed manifests in `OUT_ROOT/manifests/...`.

> W&B is **off** unless `WANDB_API_KEY` is exported (project
> `spur-da2ft-depth-ablation`). The original author's key was removed from these
> scripts — use your own.
