"""
simple_stereo/train.py
======================
Training script for StereoDepthCNN and StereoDepthDINO.

Usage examples:
    # CNN with cross-view fusion
    python MVP_MODEL/simple_stereo/train.py \
        --model cnn \
        --train_manifest manifests/.../stereo_train_boxfam.csv \
        --val_manifest   manifests/.../stereo_val_boxfam.csv \
        --out_dir checkpoints/simple_stereo_cnn

    # CNN without fusion (monocular baseline)
    python MVP_MODEL/simple_stereo/train.py \
        --model cnn --no_fusion ...

    # DINO with fusion
    python MVP_MODEL/simple_stereo/train.py \
        --model dino ...

    # DINO without fusion
    python MVP_MODEL/simple_stereo/train.py \
        --model dino --no_fusion ...
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Allow running from repo root
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from dataset.trunk_stereo_mvp import TrunkStereoMVPDataset
from MVP_MODEL.simple_stereo.model import StereoDepthCNN, StereoDepthDINO, silog_loss, rmse


# =============================================================================
# Helpers
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    args,
    scaler=None,
) -> dict:
    model.train(train)
    total_loss = 0.0
    total_rmse = 0.0
    n_batches  = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            d_pro = batch['d_pro'].to(device)   # (B, 2, 1, H, W)
            d_gt  = batch['d_gt'].to(device)    # (B, 2, 1, H, W)
            mask  = batch['mask'].to(device)    # (B, 2, 1, H, W)

            if train and scaler is not None:
                with torch.cuda.amp.autocast():
                    d_pred = model(d_pro)
                    loss, _ = silog_loss(
                        d_pred, d_gt, mask,
                        min_depth=args.min_depth, max_depth=args.max_depth,
                    )
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
            elif train:
                d_pred = model(d_pro)
                loss, _ = silog_loss(
                    d_pred, d_gt, mask,
                    min_depth=args.min_depth, max_depth=args.max_depth,
                )
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.grad_clip,
                    )
                optimizer.step()
            else:
                d_pred = model(d_pro)
                loss, _ = silog_loss(
                    d_pred, d_gt, mask,
                    min_depth=args.min_depth, max_depth=args.max_depth,
                )

            rm, _ = rmse(
                d_pred.detach(), d_gt, mask,
                min_depth=args.min_depth, max_depth=args.max_depth,
            )
            total_loss += loss.item()
            total_rmse += rm
            n_batches  += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'rmse': total_rmse / max(n_batches, 1),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    # Model selection
    ap.add_argument('--model',      choices=['cnn', 'dino'], default='cnn')
    ap.add_argument('--no_fusion',  action='store_true',
                    help='Disable cross-view fusion (monocular baseline)')
    ap.add_argument('--base',       type=int, default=64,
                    help='CNN base channel count (ignored for DINO)')

    # Data
    ap.add_argument('--train_manifest', required=True)
    ap.add_argument('--val_manifest',   required=True)
    ap.add_argument('--H',              type=int, default=280)
    ap.add_argument('--W',              type=int, default=512)
    ap.add_argument('--num_workers',    type=int, default=4)

    # Training
    ap.add_argument('--epochs',       type=int,   default=80)
    ap.add_argument('--batch_size',   type=int,   default=2)
    ap.add_argument('--lr',           type=float, default=3e-5)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--patience',     type=int,   default=10)
    ap.add_argument('--grad_clip',    type=float, default=1.0)
    ap.add_argument('--min_depth',    type=float, default=0.5)
    ap.add_argument('--max_depth',    type=float, default=10.0)
    ap.add_argument('--seed',         type=int,   default=1)
    ap.add_argument('--lr_warmup_epochs', type=int, default=5)
    ap.add_argument('--save_every',   type=int,   default=0,
                    help='Save checkpoint every N epochs (0 = best only)')

    # Output
    ap.add_argument('--out_dir', required=True)

    # W&B
    ap.add_argument('--wandb',            action='store_true')
    ap.add_argument('--wandb_project',    default='simple-stereo')
    ap.add_argument('--wandb_group',      default='')
    ap.add_argument('--wandb_run_name',   default='')

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = TrunkStereoMVPDataset(
        args.train_manifest, H=args.H, W=args.W, augment=True,
    )
    val_ds = TrunkStereoMVPDataset(
        args.val_manifest, H=args.H, W=args.W, augment=False,
    )
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model == 'cnn':
        model = StereoDepthCNN(base=args.base, no_fusion=args.no_fusion).to(device)
    else:
        model = StereoDepthDINO(no_fusion=args.no_fusion).to(device)

    total, trainable = count_params(model)
    tag = f"{args.model}{'_nofusion' if args.no_fusion else '_fusion'}"
    print(f"Seed: {args.seed}  Device: {device}  Model: {tag}")
    print(f"  {args.train_manifest}: {len(train_ds)} rows")
    print(f"  {args.val_manifest}:   {len(val_ds)} rows")
    print(f"  Params total={total:,}  trainable={trainable:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )

    def lr_lambda(epoch: int) -> float:
        if epoch < args.lr_warmup_epochs:
            return (epoch + 1) / args.lr_warmup_epochs
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # ── W&B ───────────────────────────────────────────────────────────────────
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or tag,
            name=args.wandb_run_name or f"{tag}_seed{args.seed}",
            config=vars(args),
        )
        run.watch(model, log='gradients', log_freq=50)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_ctr  = 0
    best_path     = os.path.join(args.out_dir, 'best.pt')

    for epoch in range(1, args.epochs + 1):
        tr = _run_epoch(model, train_dl, optimizer, device, train=True,
                        args=args, scaler=scaler)
        vl = _run_epoch(model, val_dl,   optimizer, device, train=False,
                        args=args, scaler=scaler)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"[{epoch:03d}/{args.epochs}]  "
            f"train loss={tr['loss']:.4f} rmse={tr['rmse']:.4f}  "
            f"val loss={vl['loss']:.4f} rmse={vl['rmse']:.4f}  "
            f"lr={lr_now:.2e}"
        )

        if run is not None:
            run.log({
                'epoch':      epoch,
                'train/loss': tr['loss'], 'train/rmse': tr['rmse'],
                'val/loss':   vl['loss'], 'val/rmse':   vl['rmse'],
                'lr':         lr_now,
            })

        # Periodic checkpoint
        if args.save_every > 0 and epoch % args.save_every == 0:
            ck = os.path.join(args.out_dir, f'epoch_{epoch:03d}.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': vl['loss']}, ck)

        # Best checkpoint + early stopping
        if vl['loss'] < best_val_loss:
            best_val_loss = vl['loss']
            patience_ctr  = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': best_val_loss}, best_path)
            print(f"  → saved best (val_loss={best_val_loss:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val_loss={best_val_loss:.4f}  saved to {best_path}")
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
