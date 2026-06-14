"""
MVP_MODEL/train_mvp_stereo_dino.py
────────────────────────────────────
Training script for MVStereoDINOUNet (DINOv2 ViT-L encoder, n_views=2 or 6).

  --n_views 2   → single stereo pair,  TrunkStereoMVPDataset
  --n_views 6   → 3 stereo pairs,      TrunkStereoTripletMVPDataset

Only trainable parameters are passed to the optimiser (DINOv2 backbone frozen).
Everything else — loss, early stopping, W&B, checkpointing — is identical to
train_mvp_stereo.py / train_mvp_stereo_triplet.py.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ── Path setup ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CV_ROOT    = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _CV_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from mvp_stereo_dino_model import (
    MVStereoDINOUNet,
    StereoDepthDINOUNet,
    silog_loss_nview,
    stereo_mv_consistency_loss,
    masked_rmse_nview,
)
from dataset.trunk_stereo_mvp         import TrunkStereoMVPDataset
from dataset.trunk_stereo_triplet_mvp import TrunkStereoTripletMVPDataset
from dataset.trunk_stereo_quad_mvp    import TrunkStereoQuadMVPDataset
from dataset.trunk_stereo_pair_mvp    import TrunkStereoPairMVPDataset


# ── W&B import ────────────────────────────────────────────────────────────────

def _import_wandb():
    for key in list(sys.modules.keys()):
        if key == "wandb" or key.startswith("wandb."):
            del sys.modules[key]

    def _has_bare_wandb(p: str) -> bool:
        root = p if p else os.getcwd()
        return os.path.isdir(os.path.join(root, "wandb")) and \
               not os.path.isfile(os.path.join(root, "wandb", "__init__.py"))

    saved   = sys.path[:]
    sys.path = [p for p in sys.path if not _has_bare_wandb(p)]
    try:
        import wandb as _w
        return _w, True
    except ImportError:
        return None, False
    finally:
        sys.path = saved


wandb, HAS_WANDB = _import_wandb()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-6) -> None:
        self.patience = patience; self.min_delta = min_delta
        self.best = float("inf"); self.wait = 0; self.should_stop = False

    def step(self, val: float) -> bool:
        if val < self.best - self.min_delta:
            self.best = val; self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
        return self.should_stop


def _aux_batch_stats(aux: dict, mask_v: torch.Tensor) -> dict:
    """One-number summary stats over the valid mask. mask_v: same shape as aux tensors."""
    eps = 1e-6
    n_valid = mask_v.sum() + eps
    out: dict = {}
    if "adaptive_delta" in aux:
        out["adaptive_delta_abs_mean"] = float(
            (aux["adaptive_delta"].abs() * mask_v).sum() / n_valid
        )
    if "delta" in aux:
        out["delta_abs_mean"] = float((aux["delta"].abs() * mask_v).sum() / n_valid)
    if "gate" in aux:
        gate = aux["gate"]
        out["gate_mean"]     = float((gate * mask_v).sum() / n_valid)
        out["gate_high_pct"] = float(((gate > 0.8).float() * mask_v).sum() / n_valid * 100.0)
        out["gate_low_pct"]  = float(((gate < 0.2).float() * mask_v).sum() / n_valid * 100.0)
    if "raw_delta" in aux:
        tsat = (aux["raw_delta"].tanh().abs() > 0.95).float()
        out["delta_saturated_pct"] = float((tsat * mask_v).sum() / n_valid * 100.0)
    return out


def _build_dummy_loader(batch_size: int, H: int, W: int, n_views: int, n: int = 8):
    class _DS(Dataset):
        def __len__(self): return n * batch_size
        def __getitem__(self, _):
            T = torch.eye(4).unsqueeze(0).repeat(n_views, 1, 1)
            for i in range(1, n_views, 2):
                T[i, 0, 3] = 0.16
            return {
                "rgb":   torch.randn(n_views, 3, H, W),
                "d_pro": torch.rand(n_views, 1, H, W).clamp(min=0.1),
                "d_gt":  torch.rand(n_views, 1, H, W).clamp(min=0.1),
                "mask":  (torch.rand(n_views, 1, H, W) > 0.5).float(),
                "K":     torch.eye(3).unsqueeze(0).repeat(n_views, 1, 1),
                "T_wc":  T,
            }
    return DataLoader(_DS(), batch_size=batch_size, shuffle=True, num_workers=0)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Train MVStereoDINOUNet — DINOv2 stereo depth refinement"
    )
    # -- data ------------------------------------------------------------------
    ap.add_argument("--train_manifest", type=str, default=None)
    ap.add_argument("--val_manifest",   type=str, default=None)
    ap.add_argument("--path_remap",     type=str, default=None)
    ap.add_argument("--H",              type=int, default=280)
    ap.add_argument("--W",              type=int, default=512)
    ap.add_argument("--num_workers",    type=int, default=4)
    ap.add_argument("--depth_only",     action="store_true",
                    help="Use StereoDepthDINOUNet: depth-only input, no DINOv2 backbone.")
    ap.add_argument("--rgb_only",       action="store_true",
                    help="Use MVStereoDINOUNet with RGB input only; "
                         "PRO depth is zeroed out so the model sees no depth signal.")
    ap.add_argument("--use_plucker",   action="store_true",
                    help="Add Plücker ray embeddings (from K and T_wc) to encoder "
                         "skip connections and bottleneck before cross-view fusion.")
    ap.add_argument("--no_pro_calib",  action="store_true",
                    help="Disable the global alpha,beta PRO depth calibration in the loader; "
                         "feed raw PRO depth (clamped to >=1e-3 m) to the encoder.")
    ap.add_argument("--unfreeze_last_n", type=int, default=0,
                    help="If >0, mark the last N DINOv2 ViT-L blocks as trainable "
                         "(partial fine-tuning of the backbone). Default 0 = fully frozen.")
    ap.add_argument("--lora_rank",       type=int, default=0,
                    help="If >0, wrap every DINOv2 attn.qkv with a LoRA adapter of "
                         "this rank (base weights stay frozen). Default 0 = no LoRA.")
    ap.add_argument("--lora_alpha",      type=float, default=16.0,
                    help="LoRA scaling alpha (only used when --lora_rank > 0).")

    # -- prediction formulation -----------------------------------------------
    ap.add_argument("--pred_mode", type=str, default="absolute",
                    choices=["absolute", "bounded_residual", "adaptive_residual"],
                    help="absolute: standard softplus depth output. "
                         "bounded_residual: pred = d_pro + max_delta * tanh(raw). "
                         "adaptive_residual: pred = d_pro + sigmoid(gate)*max_delta*tanh(delta) "
                         "(decoder head becomes 2-channel).")
    ap.add_argument("--max_delta", type=float, default=1.0,
                    help="Max correction magnitude in metres for residual modes.")

    # -- model -----------------------------------------------------------------
    ap.add_argument("--n_views",   type=int,  default=2, choices=(2, 4, 6, 8),
                    help="2 = single stereo pair, 6 = 3 stereo pairs")
    ap.add_argument("--pose_ch",   type=int,  default=32)
    ap.add_argument("--no_fusion", action="store_true",
                    help="Disable cross-view bottleneck fusion and pose encoding; "
                         "each view decoded independently (ablation baseline).")
    ap.add_argument("--no_pose",   action="store_true",
                    help="Disable pose embedding at bottleneck; views fused without "
                         "camera geometry.")

    # -- training --------------------------------------------------------------
    ap.add_argument("--epochs",       type=int,   default=80)
    ap.add_argument("--batch_size",   type=int,   default=2)
    ap.add_argument("--lr",           type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed",         type=int,   default=0)

    # -- early stopping --------------------------------------------------------
    # -- depth validity gate ---------------------------------------------------
    ap.add_argument("--min_depth", type=float, default=0.5,
                    help="Min valid GT depth (m); pixels below excluded from loss/RMSE.")
    ap.add_argument("--max_depth", type=float, default=10.0,
                    help="Max valid GT depth (m); pixels above excluded from loss/RMSE.")

    ap.add_argument("--patience", type=int, default=10)

    # -- gradient clipping + LR warmup ----------------------------------------
    ap.add_argument("--grad_clip",        type=float, default=1.0,
                    help="max-norm for gradient clipping (0 = disabled)")
    ap.add_argument("--lr_warmup_epochs", type=int,   default=5,
                    help="Linear LR warmup over first N epochs (0 = disabled)")

    # -- multi-view loss -------------------------------------------------------
    ap.add_argument("--lambda_mv",        type=float, default=0.0)
    ap.add_argument("--mv_warmup_epochs", type=int,   default=2)

    # -- checkpointing ---------------------------------------------------------
    ap.add_argument("--out_dir",    type=str, default="./checkpoints/mvp_stereo_dino")
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--resume",     type=str, default=None)

    # -- wandb -----------------------------------------------------------------
    ap.add_argument("--wandb",          action="store_true")
    ap.add_argument("--wandb_project",  type=str, default="mvp_stereo_dino")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_group",    type=str, default="dino_5seeds")

    args = ap.parse_args()
    N_VIEWS = args.n_views

    seed_out_dir = os.path.join(args.out_dir, f"seed_{args.seed:02d}")
    os.makedirs(seed_out_dir, exist_ok=True)

    run_name = args.wandb_run_name or f"dino_v{N_VIEWS}_seed{args.seed}"
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Seed: {args.seed}  Device: {device}  n_views: {N_VIEWS}")

    # -- DataLoaders -----------------------------------------------------------
    if args.train_manifest:
        if N_VIEWS == 2:
            train_ds = TrunkStereoMVPDataset(
                args.train_manifest, H=args.H, W=args.W,
                random_swap=False, path_remap=args.path_remap,
                pro_calib=not args.no_pro_calib,
            )
        elif N_VIEWS == 4:
            train_ds = TrunkStereoPairMVPDataset(
                args.train_manifest, H=args.H, W=args.W,
                path_remap=args.path_remap,
                pro_calib=not args.no_pro_calib,
            )
        elif N_VIEWS == 6:
            train_ds = TrunkStereoTripletMVPDataset(
                args.train_manifest, H=args.H, W=args.W,
                path_remap=args.path_remap,
                pro_calib=not args.no_pro_calib,
            )
        elif N_VIEWS == 8:
            train_ds = TrunkStereoQuadMVPDataset(
                args.train_manifest, H=args.H, W=args.W,
                path_remap=args.path_remap,
                pro_calib=not args.no_pro_calib,
            )
        else:
            raise ValueError(f"Unsupported n_views={N_VIEWS}; expected 2, 4, 6, or 8.")
        train_dl = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
        )
        val_dl = None
        if args.val_manifest:
            if N_VIEWS == 2:
                val_ds = TrunkStereoMVPDataset(
                    args.val_manifest, H=args.H, W=args.W,
                    random_swap=False, path_remap=args.path_remap,
                    pro_calib=not args.no_pro_calib,
                )
            elif N_VIEWS == 4:
                val_ds = TrunkStereoPairMVPDataset(
                    args.val_manifest, H=args.H, W=args.W,
                    path_remap=args.path_remap,
                    pro_calib=not args.no_pro_calib,
                )
            elif N_VIEWS == 6:
                val_ds = TrunkStereoTripletMVPDataset(
                    args.val_manifest, H=args.H, W=args.W,
                    path_remap=args.path_remap,
                    pro_calib=not args.no_pro_calib,
                )
            elif N_VIEWS == 8:
                val_ds = TrunkStereoQuadMVPDataset(
                    args.val_manifest, H=args.H, W=args.W,
                    path_remap=args.path_remap,
                    pro_calib=not args.no_pro_calib,
                )
            val_dl = DataLoader(
                val_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
    else:
        print("No --train_manifest given; running on SYNTHETIC data.")
        train_dl = _build_dummy_loader(args.batch_size, args.H, args.W, N_VIEWS)
        val_dl   = None

    # -- Model -----------------------------------------------------------------
    if args.depth_only:
        model = StereoDepthDINOUNet(
            n_views=N_VIEWS,
            no_fusion=args.no_fusion,
        ).to(device)
        trainable = list(model.parameters())   # all trainable — no frozen backbone
        print(f"StereoDepthDINOUNet(n_views={N_VIEWS}, no_fusion={args.no_fusion})  |  "
              f"{sum(p.numel() for p in trainable):,} params (all trainable)")
    else:
        model = MVStereoDINOUNet(
            n_views=N_VIEWS,
            use_pose=not args.no_pose,
            pose_ch=args.pose_ch,
            no_fusion=args.no_fusion,
            use_plucker=args.use_plucker,
            unfreeze_last_n=args.unfreeze_last_n,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            pred_mode=args.pred_mode,
            max_delta=args.max_delta,
        ).to(device)
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_total = sum(p.numel() for p in model.parameters())
        print(f"MVStereoDINOUNet(n_views={N_VIEWS})  |  "
              f"{n_total:,} total params  ({sum(p.numel() for p in trainable):,} trainable, "
              f"{n_total - sum(p.numel() for p in trainable):,} frozen DINOv2)")

    opt = optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda ep: min(1.0, (ep + 1) / args.lr_warmup_epochs)
        )
    else:
        warmup_sched = None

    start_epoch = 0
    best_rmse   = float("inf")
    best_ckpt   = None

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0)
        best_rmse   = state.get("best_rmse", float("inf"))
        print(f"Resumed  epoch={start_epoch}  best_rmse={best_rmse:.6f}")

    early_stopper = EarlyStopping(patience=args.patience) if args.patience > 0 else None

    # -- W&B -------------------------------------------------------------------
    use_wandb = HAS_WANDB and args.wandb
    if args.wandb and not HAS_WANDB:
        print("Warning: wandb not installed — continuing without it.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project, name=run_name, group=args.wandb_group,
            config={k: v for k, v in vars(args).items() if k != "wandb"},
            resume="allow" if args.resume else None,
        )
        wandb.watch(model, log="gradients", log_freq=100)

    # Build view labels: L/R for n_views=2, L1/R1/L2/R2/... for n_views=6
    if N_VIEWS == 2:
        view_labels = ["L", "R"]
    else:
        view_labels = [lbl for p in range(N_VIEWS // 2) for lbl in (f"L{p+1}", f"R{p+1}")]

    # -- Training loop ---------------------------------------------------------
    def _run_epoch(loader: DataLoader, train: bool, mv_active: bool) -> dict:
        model.train(train)
        loss_sum = mv_sum = 0.0
        pv_sum   = [0.0] * N_VIEWS
        rmse_sum = [0.0] * N_VIEWS
        aux_sum: dict = {}
        aux_n    = 0
        n = 0
        tag = "train" if train else "val"

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch_idx, batch in enumerate(loader):
                if batch_idx % 100 == 0:
                    print(f"  [{tag}] batch {batch_idx}/{len(loader)}", flush=True)

                d_pro = batch["d_pro"].to(device)
                d_gt  = batch["d_gt"].to(device)
                mask  = batch["mask"].to(device)
                K     = batch["K"].to(device)
                T_wc  = batch["T_wc"].to(device)

                if args.depth_only:
                    d_pred = model(d_pro)
                else:
                    rgb = batch["rgb"].to(device)
                    if args.rgb_only:
                        d_pro = torch.zeros_like(d_pro)
                    d_pred = model(rgb, d_pro, T_wc, K)

                if train and not torch.all(torch.isfinite(d_pred)):
                    print(f"  WARNING: non-finite prediction at batch {batch_idx}, resetting BN and skipping",
                          flush=True)
                    for m in model.modules():
                        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                            m.reset_running_stats()
                    opt.zero_grad()
                    continue

                silog_total, per_view = silog_loss_nview(
                    d_pred, d_gt, mask,
                    min_depth=args.min_depth, max_depth=args.max_depth,
                )

                mv_loss = torch.tensor(0.0, device=device)
                if mv_active and args.lambda_mv > 0:
                    mv_loss = stereo_mv_consistency_loss(d_pred, d_gt, mask, K, T_wc)
                    silog_total = silog_total + args.lambda_mv * mv_loss

                if train:
                    if not torch.isfinite(silog_total):
                        print(f"  WARNING: non-finite loss at batch {batch_idx}, resetting BN and skipping",
                              flush=True)
                        for m in model.modules():
                            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                                m.reset_running_stats()
                        opt.zero_grad()
                        continue
                    opt.zero_grad()
                    silog_total.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                    opt.step()
                    if any(not torch.all(torch.isfinite(p.data))
                           for p in model.parameters()):
                        print(f"  WARNING: weights corrupted at batch {batch_idx}, "
                              f"restoring checkpoint", flush=True)
                        _restore = best_ckpt if (best_ckpt and os.path.isfile(best_ckpt)) else _init_ckpt
                        _s = torch.load(_restore, map_location=device)
                        model.load_state_dict(_s["model"])
                        opt.load_state_dict(_s["optimizer"])
                        opt.zero_grad()
                        continue

                with torch.no_grad():
                    _, rmse_pv = masked_rmse_nview(
                        d_pred, d_gt, mask,
                        min_depth=args.min_depth, max_depth=args.max_depth,
                    )

                    aux = getattr(model, "_last_aux", None)
                    if aux is not None:
                        mask_v = (mask.float()
                                  * (d_gt > args.min_depth).float()
                                  * (d_gt < args.max_depth).float())
                        stats = _aux_batch_stats(aux, mask_v)
                        for k, val in stats.items():
                            aux_sum[k] = aux_sum.get(k, 0.0) + val
                        aux_n += 1

                loss_sum += silog_total.item()
                mv_sum   += mv_loss.item()
                for v in range(N_VIEWS):
                    pv_sum[v]   += per_view[v].item()
                    rmse_sum[v] += rmse_pv[v]
                n += 1

        nb = max(n, 1)
        ab = max(aux_n, 1)
        return {
            "loss":    loss_sum / nb,
            "mv_loss": mv_sum / nb,
            "pv":      [pv_sum[v]   / nb for v in range(N_VIEWS)],
            "rmse":    [rmse_sum[v] / nb for v in range(N_VIEWS)],
            "aux":     {k: v / ab for k, v in aux_sum.items()},
        }

    def _save(path: str) -> None:
        # Write to a local temp file first, then move — avoids partial writes on NFS.
        tmp = path + ".tmp"
        torch.save({
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "epoch": epoch + 1, "best_rmse": best_rmse, "args": vars(args),
        }, tmp)
        shutil.move(tmp, path)

    # Save initial weights so weight-corruption recovery always has a fallback
    _init_ckpt = os.path.join(seed_out_dir, "init_epoch_0000.pt")
    _init_tmp  = _init_ckpt + ".tmp"
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "epoch": start_epoch, "best_rmse": float("inf"), "args": vars(args),
    }, _init_tmp)
    shutil.move(_init_tmp, _init_ckpt)

    # -- Epoch loop ------------------------------------------------------------
    for epoch in range(start_epoch, start_epoch + args.epochs):
        mv_active = (epoch >= start_epoch + args.mv_warmup_epochs)

        tr      = _run_epoch(train_dl, train=True,  mv_active=mv_active)
        tr_rmse = sum(tr["rmse"]) / N_VIEWS

        log = {
            "epoch": epoch + 1, "seed": args.seed,
            "pred_mode": args.pred_mode, "max_delta": args.max_delta,
            "train/loss": tr["loss"], "train/mv_loss": tr["mv_loss"],
            "train/rmse": tr_rmse,
        }
        for v, lbl in enumerate(view_labels):
            log[f"train/silog_{lbl}"] = tr["pv"][v]
            log[f"train/rmse_{lbl}"]  = tr["rmse"][v]
        for k, v in tr["aux"].items():
            log[f"train/{k}"] = v

        rmse_str = "  ".join(f"{lbl}={tr['rmse'][v]:.4f}" for v, lbl in enumerate(view_labels))
        line = (f"Epoch {epoch+1:04d}  seed={args.seed}"
                f"  train: loss={tr['loss']:.6f}  mv={tr['mv_loss']:.4f}"
                f"  rmse={tr_rmse:.4f}  [{rmse_str}]")

        monitor = tr_rmse

        if val_dl is not None:
            vl      = _run_epoch(val_dl, train=False, mv_active=mv_active)
            vl_rmse = sum(vl["rmse"]) / N_VIEWS
            monitor = vl_rmse

            vl_str = "  ".join(f"{lbl}={vl['rmse'][v]:.4f}" for v, lbl in enumerate(view_labels))
            line += (f"  | val: loss={vl['loss']:.6f}  rmse={vl_rmse:.4f}  [{vl_str}]")
            log.update({
                "val/loss": vl["loss"], "val/mv_loss": vl["mv_loss"], "val/rmse": vl_rmse,
            })
            for v, lbl in enumerate(view_labels):
                log[f"val/silog_{lbl}"] = vl["pv"][v]
                log[f"val/rmse_{lbl}"]  = vl["rmse"][v]
            for k, v in vl["aux"].items():
                log[f"val/{k}"] = v

        print(line)
        if use_wandb:
            wandb.log(log, step=epoch + 1)

        if monitor < best_rmse:
            best_rmse = monitor
            if best_ckpt and os.path.isfile(best_ckpt):
                os.remove(best_ckpt)
            best_ckpt = os.path.join(seed_out_dir, f"best_epoch_{epoch+1:04d}.pt")
            _save(best_ckpt)
            print(f"  -> new best RMSE ({monitor:.4f}), saved {best_ckpt}")
            if use_wandb:
                wandb.run.summary["best_rmse"]  = best_rmse
                wandb.run.summary["best_epoch"] = epoch + 1

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            _save(os.path.join(seed_out_dir, f"epoch_{epoch+1:04d}.pt"))

        if warmup_sched is not None:
            warmup_sched.step()

        if early_stopper is not None and early_stopper.step(monitor):
            print(f"  Early stopping at epoch {epoch+1} (best={early_stopper.best:.4f})")
            if use_wandb:
                wandb.run.summary["stopped_epoch"] = epoch + 1
            break

    # Free the init snapshot — only needed mid-training as a BN-reset fallback.
    # Leaving it behind leaks ~1.3 GB per run and refills the disk.
    if os.path.isfile(_init_ckpt):
        os.remove(_init_ckpt)

    print(f"\nDone (seed={args.seed}). Best checkpoint: {best_ckpt}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
