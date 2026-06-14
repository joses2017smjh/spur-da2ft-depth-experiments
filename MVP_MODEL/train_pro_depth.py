"""
MVP_MODEL/train_pro_depth.py
─────────────────────────────
Train ProDepthUNet: stereo depth refinement using PRO depth only (no RGB, no pose).

Inputs : PRO depth pair (L, R)  — raw sensor depth from pro_refine/
Outputs: Refined depth map for each frame

Model  : MVStereoUNet(use_rgb=False, use_pose=False, no_fusion=<flag>)
Loss   : SiLog on trunk-masked GT depth pixels (both views)
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
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CV_ROOT    = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _CV_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from mvp_stereo_model import (
    MVStereoUNet,
    silog_loss_nview,
    masked_rmse_nview,
)
from dataset.trunk_stereo_mvp import TrunkStereoMVPDataset


# ── W&B import (avoids shadowing by local wandb/ run-log dir) ─────────────────
def _import_wandb():
    for key in list(sys.modules.keys()):
        if key == "wandb" or key.startswith("wandb."):
            del sys.modules[key]

    def _has_bare_wandb(p: str) -> bool:
        root = p if p else os.getcwd()
        wandb_dir  = os.path.join(root, "wandb")
        wandb_init = os.path.join(wandb_dir, "__init__.py")
        return os.path.isdir(wandb_dir) and not os.path.isfile(wandb_init)

    _saved   = sys.path[:]
    sys.path = [p for p in sys.path if not _has_bare_wandb(p)]
    try:
        import wandb as _wandb
        return _wandb, True
    except ImportError:
        return None, False
    finally:
        sys.path = _saved


wandb, HAS_WANDB = _import_wandb()


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-6) -> None:
        self.patience    = patience
        self.min_delta   = min_delta
        self.best        = float("inf")
        self.wait        = 0
        self.should_stop = False

    def step(self, val_metric: float) -> bool:
        if val_metric < self.best - self.min_delta:
            self.best = val_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Train ProDepthUNet — stereo depth refinement, no RGB, no pose"
    )

    # Data
    ap.add_argument("--train_manifest", type=str, required=True)
    ap.add_argument("--val_manifest",   type=str, default=None)
    ap.add_argument("--path_remap",     type=str, default=None,
                    help="'OLD_PREFIX:NEW_PREFIX' to fix manifest paths on compute nodes")
    ap.add_argument("--H",              type=int, default=280)
    ap.add_argument("--W",              type=int, default=512)
    ap.add_argument("--num_workers",    type=int, default=4)
    ap.add_argument("--swap_stereo",    action="store_true",
                    help="Randomly swap L/R at sample time during training")

    # Model
    ap.add_argument("--n_views",   type=int,  default=2,
                    help="Number of stereo views (default 2 = L + R)")
    ap.add_argument("--base",      type=int,  default=64,
                    help="UNet base channel width")
    ap.add_argument("--no_fusion", action="store_true",
                    help="Disable cross-view bottleneck fusion — each view decoded independently")

    # Depth validity gate
    ap.add_argument("--min_depth", type=float, default=0.5)
    ap.add_argument("--max_depth", type=float, default=10.0)

    # Training
    ap.add_argument("--epochs",        type=int,   default=80)
    ap.add_argument("--batch_size",    type=int,   default=2)
    ap.add_argument("--lr",            type=float, default=3e-4)
    ap.add_argument("--weight_decay",  type=float, default=1e-4)
    ap.add_argument("--seed",          type=int,   default=0)
    ap.add_argument("--patience",      type=int,   default=10,
                    help="Early-stopping patience (0 = disabled)")
    ap.add_argument("--grad_clip",     type=float, default=1.0,
                    help="Max gradient norm (0 = disabled)")
    ap.add_argument("--lr_warmup_epochs", type=int, default=5,
                    help="Linear LR warmup over first N epochs (0 = disabled)")

    # Checkpointing
    ap.add_argument("--out_dir",    type=str, default="./checkpoints/pro_depth")
    ap.add_argument("--save_every", type=int, default=0,
                    help="Save a periodic checkpoint every N epochs (0 = best-only)")
    ap.add_argument("--resume",     type=str, default=None)

    # W&B
    ap.add_argument("--wandb",          action="store_true")
    ap.add_argument("--wandb_project",  type=str, default="pro_depth_refine")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_group",    type=str, default="pro_depth")

    args = ap.parse_args()

    seed_out_dir = os.path.join(args.out_dir, f"seed_{args.seed:02d}")
    os.makedirs(seed_out_dir, exist_ok=True)

    run_name = args.wandb_run_name or f"pro_depth_seed{args.seed}"
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Seed: {args.seed}  Device: {device}")
    print(f"Mode: {'no-fusion' if args.no_fusion else 'cross-view fusion'}  n_views={args.n_views}")

    # ── Datasets ────────────────────────────────────────────────────────────
    print("Building datasets …")
    train_ds = TrunkStereoMVPDataset(
        args.train_manifest, H=args.H, W=args.W,
        random_swap=args.swap_stereo, path_remap=args.path_remap,
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_dl = None
    if args.val_manifest:
        val_ds = TrunkStereoMVPDataset(
            args.val_manifest, H=args.H, W=args.W,
            random_swap=False, path_remap=args.path_remap,
        )
        val_dl = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    if len(train_ds) == 0:
        sys.exit("[ERROR] Training dataset is empty — check manifest and PRO depth paths.")

    # ── Model ────────────────────────────────────────────────────────────────
    # Depth-only: use_rgb=False, use_pose=False
    model = MVStereoUNet(
        n_views=args.n_views,
        use_rgb=False,
        use_pose=False,
        base=args.base,
        no_fusion=args.no_fusion,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"ProDepthUNet  |  {n_params:,} parameters")

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_sched = None
    if args.lr_warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda ep: min(1.0, (ep + 1) / args.lr_warmup_epochs)
        )

    start_epoch = 0
    best_rmse   = float("inf")
    best_ckpt   = None

    if args.resume:
        print(f"Resuming from {args.resume}")
        state       = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch = state.get("epoch", 0)
        best_rmse   = state.get("best_rmse", float("inf"))
        print(f"  epoch={start_epoch}  best_rmse={best_rmse:.6f}")

    early_stopper = EarlyStopping(patience=args.patience) if args.patience > 0 else None

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = HAS_WANDB and args.wandb
    if args.wandb and not HAS_WANDB:
        print("Warning: wandb not installed — continuing without it.")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=args.wandb_group,
            config={k: v for k, v in vars(args).items() if k != "wandb"},
            resume="allow" if args.resume else None,
        )

    # Save initial weights so weight-corruption recovery always has a fallback
    _init_ckpt = os.path.join(seed_out_dir, "init_epoch_0000.pt")
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "epoch": start_epoch, "best_rmse": float("inf"), "args": vars(args),
    }, _init_ckpt + ".tmp")
    shutil.move(_init_ckpt + ".tmp", _init_ckpt)

    # ── Epoch runner ─────────────────────────────────────────────────────────
    def _run_epoch(loader: DataLoader, train: bool) -> dict:
        model.train(train)
        loss_sum = 0.0
        pv_sum   = [0.0] * args.n_views
        rmse_sum = [0.0] * args.n_views
        n        = 0
        tag      = "train" if train else "val"

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch_idx, batch in enumerate(loader):
                if batch_idx % 100 == 0:
                    print(f"  [{tag}] {batch_idx}/{len(loader)}", flush=True)

                d_pro = batch["d_pro"].to(device)   # (B, 2, 1, H, W)
                d_gt  = batch["d_gt"].to(device)    # (B, 2, 1, H, W)
                mask  = batch["mask"].to(device)    # (B, 2, 1, H, W)

                # Forward — pass None for rgb and pose_T (depth-only mode)
                d_pred = model(rgb=None, d_pro=d_pro, pose_T=None)  # (B, 2, 1, H, W)

                if train and not torch.all(torch.isfinite(d_pred)):
                    print(f"  WARNING: non-finite prediction at batch {batch_idx}, skipping",
                          flush=True)
                    opt.zero_grad()
                    continue

                loss, per_view = silog_loss_nview(
                    d_pred, d_gt, mask,
                    min_depth=args.min_depth, max_depth=args.max_depth,
                )

                if train:
                    if not torch.isfinite(loss):
                        print(f"  WARNING: non-finite loss at batch {batch_idx}, skipping",
                              flush=True)
                        opt.zero_grad()
                        continue
                    opt.zero_grad()
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()

                with torch.no_grad():
                    _, rmse_pv = masked_rmse_nview(
                        d_pred, d_gt, mask,
                        min_depth=args.min_depth, max_depth=args.max_depth,
                    )

                loss_sum += loss.item()
                for v in range(args.n_views):
                    pv_sum[v]   += per_view[v].item()
                    rmse_sum[v] += rmse_pv[v]
                n += 1

        nb = max(n, 1)
        return {
            "loss": loss_sum / nb,
            "pv":   [pv_sum[v]   / nb for v in range(args.n_views)],
            "rmse": [rmse_sum[v] / nb for v in range(args.n_views)],
        }

    def _save(path: str) -> None:
        tmp = path + ".tmp"
        torch.save({
            "model":     model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch":     epoch + 1,
            "best_rmse": best_rmse,
            "args":      vars(args),
        }, tmp)
        shutil.move(tmp, path)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + args.epochs):
        tr      = _run_epoch(train_dl, train=True)
        tr_rmse = sum(tr["rmse"]) / args.n_views

        log = {
            "epoch":         epoch + 1,
            "seed":          args.seed,
            "train/loss":    tr["loss"],
            "train/rmse":    tr_rmse,
        }
        for v in range(args.n_views):
            log[f"train/silog_v{v}"] = tr["pv"][v]
            log[f"train/rmse_v{v}"]  = tr["rmse"][v]

        line = (f"Epoch {epoch+1:04d}  seed={args.seed}"
                f"  train: loss={tr['loss']:.6f}  rmse={tr_rmse:.4f}"
                f"  [{' '.join(f'v{v}={tr[\"rmse\"][v]:.4f}' for v in range(args.n_views))}]")

        monitor = tr_rmse

        if val_dl is not None:
            vl      = _run_epoch(val_dl, train=False)
            vl_rmse = sum(vl["rmse"]) / args.n_views
            monitor = vl_rmse

            line += (f"  | val: loss={vl['loss']:.6f}  rmse={vl_rmse:.4f}"
                     f"  [{' '.join(f'v{v}={vl[\"rmse\"][v]:.4f}' for v in range(args.n_views))}]")

            log.update({
                "val/loss": vl["loss"],
                "val/rmse": vl_rmse,
            })
            for v in range(args.n_views):
                log[f"val/silog_v{v}"] = vl["pv"][v]
                log[f"val/rmse_v{v}"]  = vl["rmse"][v]

        print(line)

        if use_wandb:
            try:
                wandb.log(log, step=epoch + 1)
            except Exception as e:
                print(f"[WARN] wandb.log failed: {e}", flush=True)

        if monitor < best_rmse:
            best_rmse = monitor
            if best_ckpt and os.path.isfile(best_ckpt):
                os.remove(best_ckpt)
            best_ckpt = os.path.join(seed_out_dir, f"best_epoch_{epoch+1:04d}.pt")
            _save(best_ckpt)
            print(f"  -> new best RMSE ({monitor:.4f}), saved {best_ckpt}")
            if use_wandb:
                try:
                    wandb.run.summary["best_rmse"]  = best_rmse
                    wandb.run.summary["best_epoch"] = epoch + 1
                except Exception:
                    pass

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            _save(os.path.join(seed_out_dir, f"epoch_{epoch+1:04d}.pt"))

        if warmup_sched is not None:
            warmup_sched.step()

        if early_stopper is not None:
            if early_stopper.step(monitor):
                print(f"  Early stopping at epoch {epoch+1} "
                      f"(patience={args.patience}, best={early_stopper.best:.4f})")
                if use_wandb:
                    try:
                        wandb.run.summary["stopped_epoch"] = epoch + 1
                    except Exception:
                        pass
                break

    print(f"\nDone (seed={args.seed}). Best checkpoint: {best_ckpt}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
