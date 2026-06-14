#!/usr/bin/env python3
"""
train_depth_da2.py
──────────────────
Fine-tune Depth Anything V2 (metric) on orchard trunk images.

This script mirrors the official metric_depth/train.py as closely as possible:
  github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth

Differences from the official script (documented inline):
  - Distributed training removed (single GPU only).
  - TensorBoard replaced with W&B.
  - Dataset: TrunkDA2 (manifest CSV) instead of Hypersim / VKITTI2.
  - Checkpoint saves best.pth by val RMSE (official only saves latest.pth).
  - --resume support added.

Everything else — loss, optimizer param grouping, polynomial LR decay,
total_iters computation, training loop structure, validation loop structure,
metric computation, previous_best tracking — is a direct copy of the
official train.py.

Setup
─────
  conda activate depth-env

  # Make the official DA2 code importable:
  export PYTHONPATH=/path/to/Depth-Anything-V2:$PYTHONPATH

  # Download pretrained weights (vitl example):
  huggingface-cli download depth-anything/Depth-Anything-V2-Large \
      --local-dir /path/to/da2_weights

Usage
─────
  python train_depth_da2.py \\
      --encoder vitl \\
      --pretrained-from /path/to/da2_weights/depth_anything_v2_vitl.pth \\
      --train-manifest /nfs/hpc/share/sanchej7/Computer_Vision/train_manifest.csv \\
      --val-manifest   /nfs/hpc/share/sanchej7/Computer_Vision/val_manifest.csv \\
      --save-path      /nfs/hpc/share/sanchej7/Computer_Vision/checkpoints/da2_metric
"""

import argparse
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb

from dataset.trunk_da2 import TrunkDA2

# ──────────────────────────────────────────────────────────────────────────────
# SiLogLoss — verbatim copy of metric_depth/util/loss.py
# ──────────────────────────────────────────────────────────────────────────────

class SiLogLoss(nn.Module):
    """
    Scale-invariant log loss.
    Verbatim from metric_depth/util/loss.py.
    """

    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        # valid_mask is detached so gradients do not flow through it
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean()
            - self.lambd * torch.pow(diff_log.mean(), 2)
        )
        return loss


# ──────────────────────────────────────────────────────────────────────────────
# eval_depth — verbatim copy of metric_depth/util/metric.py
# ──────────────────────────────────────────────────────────────────────────────

def eval_depth(pred, target):
    """
    Standard depth evaluation metrics.
    Verbatim from metric_depth/util/metric.py.

    Args:
        pred   : 1-D tensor of predicted depths at valid pixels.
        target : 1-D tensor of GT metric depths at valid pixels.

    Returns dict with keys:
        d1, d2, d3, abs_rel, sq_rel, rmse, rmse_log, log10, silog
    """
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25     ).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff     = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel  = torch.mean(torch.abs(diff) / target)
    sq_rel   = torch.mean(torch.pow(diff, 2) / target)
    rmse     = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))
    log10    = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    # Note: silog in eval_depth hardcodes 0.5; SiLogLoss uses self.lambd.
    silog    = torch.sqrt(
        torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)
    )

    return {
        'd1':       d1.item(),
        'd2':       d2.item(),
        'd3':       d3.item(),
        'abs_rel':  abs_rel.item(),
        'sq_rel':   sq_rel.item(),
        'rmse':     rmse.item(),
        'rmse_log': rmse_log.item(),
        'log10':    log10.item(),
        'silog':    silog.item(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Model configs — verbatim copy from official train.py
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser — mirrors official train.py argument names exactly
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='Depth Anything V2 for Metric Depth Estimation',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# Mirrors official args exactly:
parser.add_argument('--encoder',   default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--img-size',  default=518,    type=int)
parser.add_argument('--min-depth', default=0.001,  type=float)
parser.add_argument('--max-depth', default=20,     type=float)
parser.add_argument('--epochs',    default=40,     type=int)
parser.add_argument('--bs',        default=2,      type=int)
parser.add_argument('--lr',        default=0.000005, type=float,
                    help='Base LR for backbone. DPT head gets lr×10. '
                         'Official default: 5e-6.')
parser.add_argument('--pretrained-from', type=str, required=True,
                    help='Path to pretrained DA2 .pth weights file. '
                         'Official loading: only keys where "pretrained" in name '
                         'are loaded, strict=False (backbone only).')
parser.add_argument('--save-path', type=str, required=True)

# Our additions (not in official script):
parser.add_argument('--train-manifest', type=str, required=True,
                    help='Path to train_manifest.csv')
parser.add_argument('--val-manifest',   type=str, required=True,
                    help='Path to val_manifest.csv')
parser.add_argument('--num-workers', default=4,  type=int)
parser.add_argument('--seed',        default=42, type=int)
parser.add_argument('--resume',      type=str,   default=None,
                    help='Path to a checkpoint .pth to resume from.')
parser.add_argument('--curriculum', action='store_true',
                    help='Enable epoch-based curriculum sampling. '
                         'epoch 0-1: base images only. '
                         'epoch >=2: 30%% base / 70%% cam. '
                         'Requires set_id column in the manifest CSV.')
parser.add_argument('--wandb-project',  default='trunk-depth-da2-metric')
parser.add_argument('--wandb-run-name', default=None)
parser.add_argument('--eval-box-mask', action='store_true',
                    help='Also compute SiLog on the tight bounding-box region of '
                         'the trunk mask during validation (logged as '
                         'eval/silog_trunk and eval/silog_box).')
parser.add_argument('--train-box-mask', action='store_true',
                    help='Expand training loss mask to include the box mask region '
                         '(union of trunk mask and box mask). Without this flag only '
                         'trunk-masked pixels contribute to loss.')
parser.add_argument('--box-dilation-px', type=int, default=8,
                    help='Radius (px) used to dilate the trunk silhouette into the '
                         'box mask. Default 8. Set 0 to make box_mask == trunk_mask. '
                         'Replaces the legacy bbox-of-silhouette construction, which '
                         'flooded the loss with ~73%% background pixels.')
parser.add_argument('--box-mask-global', type=str, default='',
                    help='Path to a single binary PNG used as the box_mask for every '
                         'training sample (overrides dilation). Same file is applied '
                         'to all frames, resized + cropped alongside the image.')
parser.add_argument('--box-loss-mode', choices=['union', 'weighted', 'balanced', 'anchor'],
                    default='union',
                    help='How to combine trunk and box pixels in the training loss. '
                         'union: SiLog(trunk | box) all pixels equal. '
                         'weighted: per-pixel weights rebalance trunk vs box-only by count. '
                         'balanced: 0.5*SiLog(trunk) + 0.5*SiLog(box_only). '
                         'anchor: SiLog(trunk) + lambda*L1(box_only).')
parser.add_argument('--box-loss-lambda', type=float, default=0.2,
                    help='Lambda for the scale-anchor mode (weight of the L1 term on '
                         'box-only pixels). Only used when --box-loss-mode=anchor.')
parser.add_argument('--max-runtime-seconds', type=int, default=0,
                    help='Wall-clock training budget. After each epoch finishes '
                         '(post-validation, post-checkpoint save) the script '
                         'checks elapsed time and exits cleanly if >= this value. '
                         '0 disables the check (run all --epochs).')


# ──────────────────────────────────────────────────────────────────────────────
# Main — structure mirrors official train.py exactly
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parser.parse_args()

    warnings.simplefilter('ignore', np.RankWarning)

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled   = True
    cudnn.benchmark = True

    os.makedirs(args.save_path, exist_ok=True)

    # ── W&B (replaces official TensorBoard SummaryWriter) ────────────────────
    # Official: writer = SummaryWriter(args.save_path)
    # We use W&B; on HPC nodes without TTY we fall back to offline mode.
    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    except Exception:
        os.environ['WANDB_MODE'] = 'offline'
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
        print('[WARN] W&B running in offline mode — no API key found.')

    # ── Datasets — mirrors official structure ─────────────────────────────────
    # Official:
    #   trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    #   valset   = Hypersim('dataset/splits/hypersim/val.txt',   'val',   size=size)
    size = (args.img_size, args.img_size)
    trainset = TrunkDA2(args.train_manifest, mode='train', size=size,
                        box_dilation_px=args.box_dilation_px,
                        global_box_mask_path=args.box_mask_global)
    valset   = TrunkDA2(args.val_manifest,   mode='val',   size=size,
                        box_dilation_px=args.box_dilation_px,
                        global_box_mask_path=args.box_mask_global)

    # Official valloader: shuffle=False, drop_last=True.
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True,
        num_workers=args.num_workers, drop_last=True, shuffle=False,
    )

    # With curriculum the trainloader is recreated each epoch, so we build it
    # inside the loop (see below). Without curriculum it is built once here.
    # In both cases total_iters uses the FULL dataset size (len(trainset) before
    # any curriculum filtering) so the LR schedule is stable across epochs.
    if not args.curriculum:
        trainloader = DataLoader(
            trainset, batch_size=args.bs, pin_memory=True,
            num_workers=args.num_workers, drop_last=True, shuffle=True,
        )

    # ── Model — verbatim from official train.py ───────────────────────────────
    # Requires: export PYTHONPATH=/path/to/Depth-Anything-V2:$PYTHONPATH
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        import sys
        sys.exit(
            "\n[ERROR] depth_anything_v2 not importable.\n"
            "Run: export PYTHONPATH=/path/to/Depth-Anything-V2:$PYTHONPATH\n"
        )

    # Official instantiation:
    #   model = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    model = DepthAnythingV2(**{**MODEL_CONFIGS[args.encoder], 'max_depth': args.max_depth})

    # Official pretrained-weight loading — backbone ('pretrained') keys only,
    # strict=False so the metric DPT head initialises from scratch:
    #   model.load_state_dict(
    #       {k: v for k, v in torch.load(args.pretrained_from, ...).items()
    #        if 'pretrained' in k}, strict=False)
    if args.pretrained_from:
        model.load_state_dict(
            {k: v
             for k, v in torch.load(args.pretrained_from, map_location='cpu').items()
             if 'pretrained' in k},
            strict=False,
        )
        print(f'Loaded pretrained backbone weights from: {args.pretrained_from}')

    # Official uses SyncBatchNorm + DDP; single-GPU skips both.
    model = model.cuda()
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    # ── Loss — official SiLogLoss ─────────────────────────────────────────────
    # Official: criterion = SiLogLoss().cuda(local_rank)
    criterion = SiLogLoss().cuda()

    # ── Optimizer — verbatim param grouping from official train.py ────────────
    # 'pretrained' in name → backbone parameters → base LR
    # 'pretrained' not in name → DPT head parameters → 10× LR
    #
    # Official:
    #   optimizer = AdamW(
    #       [{'params': [p for n,p in model.named_parameters() if 'pretrained' in n],
    #         'lr': args.lr},
    #        {'params': [p for n,p in model.named_parameters() if 'pretrained' not in n],
    #         'lr': args.lr * 10.0}],
    #       lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer = AdamW(
        [
            {
                'params': [p for n, p in model.named_parameters() if 'pretrained' in n],
                'lr':     args.lr,
            },
            {
                'params': [p for n, p in model.named_parameters() if 'pretrained' not in n],
                'lr':     args.lr * 10.0,
            },
        ],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01,
    )

    # ── total_iters ────────────────────────────────────────────────────────────
    # Official (no curriculum): total_iters = args.epochs * len(trainloader)
    # With curriculum: trainloader is recreated each epoch and its length
    # varies, so we compute from the FULL dataset size instead. This keeps the
    # LR schedule identical whether curriculum is on or off.
    total_iters = args.epochs * (len(trainset) // args.bs)

    # ── previous_best — verbatim initial values from official train.py ────────
    previous_best = {
        'd1': 0, 'd2': 0, 'd3': 0,
        'abs_rel': 100, 'sq_rel': 100, 'rmse': 100,
        'rmse_log': 100, 'log10': 100, 'silog': 100,
    }

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch   = ckpt['epoch'] + 1
        previous_best = ckpt['previous_best']
        print(f'Resumed from {args.resume}  (last completed epoch {ckpt["epoch"]})')

    # ── Training loop — mirrors official train.py exactly ────────────────────
    t_train_start = time.time()
    if args.max_runtime_seconds > 0:
        print(f'[runtime-budget] Will exit cleanly after any epoch where '
              f'elapsed >= {args.max_runtime_seconds}s '
              f'(~{args.max_runtime_seconds/3600:.2f} h).')
    for epoch in range(start_epoch, args.epochs):

        # Official: logger.info('===========> Epoch: ...')  (two lines)
        print('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(
            epoch, args.epochs,
            previous_best['d1'], previous_best['d2'], previous_best['d3']))
        print('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, '
              'rmse: {:.3f}, rmse_log: {:.3f}, log10: {:.3f}, silog: {:.3f}'.format(
                  epoch, args.epochs,
                  previous_best['abs_rel'], previous_best['sq_rel'],
                  previous_best['rmse'],    previous_best['rmse_log'],
                  previous_best['log10'],   previous_best['silog']))

        # Official: trainloader.sampler.set_epoch(epoch + 1)
        # Single-GPU: no sampler to update.

        # ── Curriculum (custom addition, not in official DA2) ─────────────────
        # Mirrors original TrunkManifest curriculum exactly:
        #   epoch 0-1 → base images only
        #   epoch >=2 → 30% base / 70% cam
        # Trainloader is recreated each epoch so the new _active index takes effect.
        if args.curriculum:
            phase = 'base-only' if epoch < 2 else '30/70 base+cam'
            trainset.set_curriculum_epoch(epoch)
            trainloader = DataLoader(
                trainset, batch_size=args.bs, pin_memory=True,
                num_workers=args.num_workers, drop_last=True, shuffle=True,
            )
            print(f'  curriculum phase: {phase}  '
                  f'({len(trainset):,} rows this epoch)')

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0

        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            # Official: img, depth, valid_mask = sample['image'].cuda(), ...
            img        = sample['image'].cuda()       # (B, 3, H, W)
            depth      = sample['depth'].cuda()       # (B, H, W)
            valid_mask = sample['valid_mask'].cuda()  # (B, H, W) float32 {0,1}
            if args.train_box_mask:
                box_mask = sample['box_mask'].cuda()  # (B, H, W) float32 {0,1}

            # Official horizontal flip augmentation (50% probability)
            if random.random() < 0.5:
                img        = img.flip(-1)
                depth      = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)
                if args.train_box_mask:
                    box_mask = box_mask.flip(-1)

            # Official forward: pred = model(img)
            # DA2 returns (B, H, W) metric depth directly — no unsqueeze needed.
            pred = model(img)  # (B, H, W)

            # Loss: 4 modes selectable via --box-loss-mode.
            #   union     : SiLog((valid|box) ∩ depth_range)
            #   weighted  : per-pixel weights rebalance trunk vs box-only by count
            #   balanced  : 0.5*SiLog(trunk) + 0.5*SiLog(box_only)
            #   anchor    : SiLog(trunk) + lambda*L1(box_only)
            depth_range = (depth >= args.min_depth) & (depth <= args.max_depth)
            trunk_m     = (valid_mask == 1) & depth_range
            if args.train_box_mask:
                box_only_m = (box_mask == 1) & (valid_mask != 1) & depth_range
            else:
                box_only_m = torch.zeros_like(trunk_m)

            mode = args.box_loss_mode if args.train_box_mask else 'union'
            if mode == 'union' or box_only_m.sum() < 10:
                # Plain SiLog on the union (or fall back to trunk-only if no box pixels).
                mask = trunk_m | box_only_m
                loss = criterion(pred, depth, mask)
            elif mode == 'weighted':
                # Weighted SiLog: pixel weights = 1 for trunk, n_trunk/n_box for box-only.
                n_t = trunk_m.sum().clamp(min=1).float()
                n_b = box_only_m.sum().clamp(min=1).float()
                w   = torch.where(trunk_m, torch.ones_like(depth),
                                  torch.full_like(depth, (n_t / n_b).item()))
                mask     = trunk_m | box_only_m
                diff_log = torch.log(depth[mask]) - torch.log(pred[mask])
                ww       = w[mask]
                sw       = ww.sum().clamp(min=1e-8)
                m1       = (ww * diff_log).sum() / sw
                m2       = (ww * diff_log.pow(2)).sum() / sw
                loss     = torch.sqrt((m2 - 0.5 * m1.pow(2)).clamp(min=1e-8))
            elif mode == 'balanced':
                loss_t = criterion(pred, depth, trunk_m) if trunk_m.sum() >= 10 else 0.0
                loss_b = criterion(pred, depth, box_only_m)
                loss   = 0.5 * loss_t + 0.5 * loss_b
            elif mode == 'anchor':
                loss_t = criterion(pred, depth, trunk_m) if trunk_m.sum() >= 10 else 0.0
                loss_b = (pred[box_only_m] - depth[box_only_m]).abs().mean()
                loss   = loss_t + args.box_loss_lambda * loss_b
            else:
                raise ValueError(f"unknown box-loss-mode: {mode}")

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Official polynomial LR decay — computed AFTER step(), takes effect
            # on the NEXT iteration:
            #   iters = epoch * len(trainloader) + i
            #   lr    = args.lr * (1 - iters / total_iters) ** 0.9
            iters = epoch * len(trainloader) + i
            lr    = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * 10.0

            # Official: writer.add_scalar('train/loss', loss.item(), iters)
            wandb.log({'train/loss': loss.item(), 'iters': iters})

            # Official: logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}')
            if i % 100 == 0:
                print('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(
                    i, len(trainloader),
                    optimizer.param_groups[0]['lr'],
                    loss.item()))

        wandb.log({
            'train/epoch_loss': total_loss / max(len(trainloader), 1),
            'epoch': epoch,
        })

        # ── Validate — mirrors official validation loop ────────────────────────
        model.eval()

        # Official accumulates results as cuda tensors for dist.reduce;
        # single-GPU uses plain Python floats.
        results = {
            k: 0.0
            for k in ('d1', 'd2', 'd3', 'abs_rel', 'sq_rel',
                      'rmse', 'rmse_log', 'log10', 'silog')
        }
        nsamples = 0
        silog_box_sum = 0.0
        nsamples_box  = 0

        for i, sample in enumerate(valloader):
            # Official: img, depth, valid_mask = sample['image'].cuda().float(), ...
            img        = sample['image'].cuda().float()  # (1, 3, H, W)
            depth      = sample['depth'].cuda()[0]       # (H, W)
            valid_mask = sample['valid_mask'].cuda()[0]  # (H, W)

            with torch.no_grad():
                pred = model(img)  # (1, H, W)
                # Official: interpolate pred to depth spatial size (safe for
                # non-square validation images that were not cropped).
                pred = F.interpolate(
                    pred[:, None], depth.shape[-2:],
                    mode='bilinear', align_corners=True,
                )[0, 0]  # (H, W)

            depth_range = (depth >= args.min_depth) & (depth <= args.max_depth)

            # Official validation mask:
            #   valid_mask = (valid_mask == 1) & (depth >= min_depth) & (depth <= max_depth)
            valid_mask = (valid_mask == 1) & depth_range

            if valid_mask.sum() < 10:
                continue

            # Official: cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

            for k in results:
                results[k] += cur_results[k]
            nsamples += 1

            # -- Box mask SiLog (opt-in via --eval-box-mask) -------------------
            if args.eval_box_mask:
                box_mask = sample['box_mask'].cuda()[0]   # (H, W)
                box_eval = (box_mask == 1) & depth_range
                if box_eval.sum() >= 10:
                    diff_log = torch.log(pred[box_eval]) - torch.log(depth[box_eval])
                    silog_box = torch.sqrt(
                        torch.pow(diff_log, 2).mean()
                        - 0.5 * torch.pow(diff_log.mean(), 2)
                    ).item()
                    silog_box_sum += silog_box
                    nsamples_box  += 1

        # Official: torch.distributed.barrier() + dist.reduce() → skipped.

        if nsamples == 0:
            print('[WARN] No valid validation samples this epoch.')
            continue

        # Official: (results[k] / nsamples).item() for each metric
        for k in results:
            results[k] /= nsamples

        # Official logging format (verbatim):
        print('==========================================================================================')
        print('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(
            *tuple(results.keys())))
        print('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(
            *tuple(results.values())))

        extra_log = {}
        if args.eval_box_mask:
            silog_box = silog_box_sum / max(nsamples_box, 1)
            print(f'  silog_trunk={results["silog"]:.4f}  silog_box={silog_box:.4f}  '
                  f'(box_n={nsamples_box}/{nsamples})')
            extra_log = {'eval/silog_trunk': results['silog'], 'eval/silog_box': silog_box}

        print('==========================================================================================')
        print()

        # Official: writer.add_scalar(f'eval/{name}', ..., epoch)
        wandb.log({**{f'eval/{k}': v for k, v in results.items()}, **extra_log, 'epoch': epoch})

        # ── Update previous_best — verbatim from official train.py ────────────
        # Official:
        #   for k in results.keys():
        #       if k in ['d1', 'd2', 'd3']:
        #           previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
        #       else:
        #           previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        for k in ('d1', 'd2', 'd3'):
            previous_best[k] = max(previous_best[k], results[k])
        for k in ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog'):
            previous_best[k] = min(previous_best[k], results[k])

        # ── Checkpoint ────────────────────────────────────────────────────────
        # Official: only saves latest.pth
        # We additionally save best.pth (by val RMSE).
        checkpoint = {
            'model':         model.state_dict(),
            'optimizer':     optimizer.state_dict(),
            'epoch':         epoch,
            'previous_best': previous_best,
        }
        def _atomic_save(path: str) -> None:
            tmp = path + '.tmp'
            torch.save(checkpoint, tmp)
            shutil.move(tmp, path)

        _atomic_save(os.path.join(args.save_path, 'latest.pth'))

        # best.pth: saved when results['rmse'] equals the updated minimum
        # (i.e., this epoch set a new RMSE record).
        if results['rmse'] == previous_best['rmse']:
            _atomic_save(os.path.join(args.save_path, 'best.pth'))
            print(f'  Saved best.pth  (rmse={results["rmse"]:.4f} m)')

        # ── Wall-clock budget check — break cleanly after a finished epoch ───
        if args.max_runtime_seconds > 0:
            elapsed = time.time() - t_train_start
            if elapsed >= args.max_runtime_seconds:
                print(f'[runtime-budget] elapsed {elapsed:.0f}s '
                      f'(~{elapsed/3600:.2f} h) >= budget '
                      f'{args.max_runtime_seconds}s after epoch {epoch}. '
                      f'Exiting cleanly. best.pth + latest.pth are preserved.')
                break

    wandb.finish()


if __name__ == '__main__':
    main()
