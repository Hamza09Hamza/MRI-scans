#!/usr/bin/env python3
"""
Train 3D U-Net for volumetric brain tumor segmentation.
Supports interrupt and resume via full checkpointing.
"""

import os
import json
import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path

from utils.unet_3d import UNet3D, count_parameters
from utils.multi_dataset_3d import create_combined_dataset


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "brats_dir": "./data/BraTS2020",
    "upenn_dir": "./data/upenn_gbm_nifti",
    "output_dir": "./segmentation_models",
    "model_name": "unet3d_multimodal",
    "device": "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    "epochs": 50,
    "batch_size": 2,          # M1 Pro: 2 | RTX 3050: 2-3 | RTX 4090: 4+
    "learning_rate": 1e-4,
    "num_workers": 2,
    "pin_memory": True,
    "modalities": ["t1ce", "flair"],
    "patch_size": (64, 128, 128),
    "num_classes": 4,
    "in_channels": 2,
    "use_gradient_checkpointing": False,  # Set True for RTX 3050
    "checkpoint_every": 1,    # Save a resume checkpoint every N epochs
    "keep_last_n": 3,         # Keep only the N most recent resume checkpoints
    "resume": "latest",       # "latest" = auto-detect | None = start fresh | path = specific file
}


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================

def save_checkpoint(state, output_dir, filename):
    """Save full training state (model + optimizer + scheduler + history)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    return path


def find_latest_checkpoint(output_dir):
    """Find the most recent resume checkpoint in output_dir."""
    checkpoints = sorted(Path(output_dir).glob("resume_epoch*.pt"))
    return str(checkpoints[-1]) if checkpoints else None


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load training state and return (start_epoch, history, best_val_dice)."""
    print(f"Resuming from: {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state["epoch"] + 1, state["history"], state["best_val_dice"]


def prune_old_checkpoints(output_dir, keep_last_n):
    """Delete old resume checkpoints, keep only the N most recent."""
    checkpoints = sorted(Path(output_dir).glob("resume_epoch*.pt"))
    for old in checkpoints[:-keep_last_n]:
        old.unlink()


# ============================================================================
# LOSS & METRICS
# ============================================================================

def dice_loss(pred, target, smooth=1.0, num_classes=4):
    """Dice loss for multi-class segmentation."""
    pred = torch.softmax(pred, dim=1)
    total_loss = 0.0
    for c in range(num_classes):
        pred_c = pred[:, c].contiguous().view(-1)
        target_c = (target == c).float().contiguous().view(-1)
        intersection = (pred_c * target_c).sum()
        dice = (2.0 * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        total_loss += 1.0 - dice
    return total_loss / num_classes


def dice_score(pred, target, num_classes=4):
    """Dice coefficient for evaluation."""
    pred = torch.argmax(pred, dim=1)
    total_dice = 0.0
    for c in range(num_classes):
        pred_c = (pred == c).float().view(-1)
        target_c = (target == c).float().view(-1)
        intersection = (pred_c * target_c).sum()
        total_dice += (2.0 * intersection) / (pred_c.sum() + target_c.sum() + 1e-5)
    return total_dice / num_classes


# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

def train_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, masks) + dice_loss(outputs, masks, num_classes=config["num_classes"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


def validate(model, loader, device, config):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    with torch.no_grad():
        pbar = tqdm(loader, desc="    Val", leave=False)
        for batch in pbar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, masks) + dice_loss(outputs, masks, num_classes=config["num_classes"])
            dice = dice_score(outputs, masks, num_classes=config["num_classes"])
            total_loss += loss.item()
            total_dice += dice.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")
    return total_loss / len(loader), total_dice / len(loader)


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    device = torch.device(CONFIG["device"])

    print(f"\n{'='*70}")
    print(f"  3D Brain Tumor Segmentation")
    print(f"{'='*70}")
    print(f"  Device    : {device}")
    print(f"  BraTS2020 : {CONFIG['brats_dir']}")
    print(f"  UPENN-GBM : {CONFIG['upenn_dir']}")
    print(f"  Modalities: {CONFIG['modalities']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Patch size: {CONFIG['patch_size']}")
    print(f"{'='*70}\n")

    # ── Data ────────────────────────────────────────────────────────────────
    print("Loading datasets...")
    combined = create_combined_dataset(
        brats_dir=CONFIG["brats_dir"],
        upenn_dir=CONFIG["upenn_dir"],
        split="train",
        modalities=CONFIG["modalities"],
        patch_size=CONFIG["patch_size"],
    )
    brats_val = create_combined_dataset(
        brats_dir=CONFIG["brats_dir"],
        upenn_dir=None,
        split="val",
        modalities=CONFIG["modalities"],
        patch_size=CONFIG["patch_size"],
    )

    n_train = int(0.85 * len(combined))
    train_set, val_split = random_split(
        combined, [n_train, len(combined) - n_train],
        generator=torch.Generator().manual_seed(42),
    )
    val_set = ConcatDataset([val_split, brats_val])

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=CONFIG["num_workers"],
                              pin_memory=CONFIG["pin_memory"])
    val_loader = DataLoader(val_set, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=CONFIG["num_workers"])

    print(f"  Train: {len(train_set)}  |  Val: {len(val_set)}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    print("Building 3D U-Net...")
    model = UNet3D(in_channels=CONFIG["in_channels"],
                   out_channels=CONFIG["num_classes"], features=32).to(device)
    print(f"  Parameters: {count_parameters(model):,}\n")

    if CONFIG.get("use_gradient_checkpointing", False):
        print("  ✓ Gradient checkpointing enabled\n")
        for m in model.modules():
            if hasattr(m, 'gradient_checkpointing'):
                m.gradient_checkpointing = True

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # ── Resume ──────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_dice = 0.0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    resume = CONFIG.get("resume", "latest")
    if resume == "latest":
        resume = find_latest_checkpoint(CONFIG["output_dir"])

    if resume and os.path.exists(resume):
        start_epoch, history, best_val_dice = load_checkpoint(
            resume, model, optimizer, scheduler, device
        )
        print(f"  Resuming at epoch {start_epoch}  |  Best Dice so far: {best_val_dice:.4f}\n")
    else:
        print("  Starting fresh training\n")

    remaining = CONFIG["epochs"] - start_epoch + 1
    if remaining <= 0:
        print(f"Already completed {CONFIG['epochs']} epochs. Increase CONFIG['epochs'] to train more.")
        return

    print(f"Training epochs {start_epoch} → {CONFIG['epochs']}  ({remaining} remaining)...")
    print(f"  Tip: Press Ctrl+C at any time — progress will be saved before exiting.\n")

    # ── Graceful interrupt ──────────────────────────────────────────────────
    interrupted = False
    def _handle_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n\n  [!] Interrupt received — will save checkpoint and exit cleanly...\n")
    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, CONFIG)
        val_loss, val_dice = validate(model, val_loader, device, CONFIG)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{CONFIG['epochs']} | "
              f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"Dice {val_dice:.4f} | LR {lr_now:.2e}")

        # Save best model weights only (lightweight — no optimizer state)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_path = os.path.join(CONFIG["output_dir"],
                                     f"{CONFIG['model_name']}_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  ↑ New best Dice {val_dice:.4f} — saved to {best_path}")

        # Periodic resume checkpoint (full state — lets you stop and continue)
        if epoch % CONFIG["checkpoint_every"] == 0 or interrupted:
            ckpt_path = save_checkpoint(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_dice": best_val_dice,
                    "history": history,
                    "config": CONFIG,
                },
                CONFIG["output_dir"],
                f"resume_epoch{epoch:03d}.pt",
            )
            prune_old_checkpoints(CONFIG["output_dir"], CONFIG["keep_last_n"])
            print(f"  ✓ Checkpoint saved → {ckpt_path}")

        # Save history after every epoch (useful for monitoring mid-run)
        with open(os.path.join(CONFIG["output_dir"], "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if interrupted:
            print(f"\n  Training paused at epoch {epoch}.")
            print(f"  Resume by running:  python3 train_3d_segmentation.py")
            print(f"  (CONFIG['resume'] = 'latest' will pick up automatically)\n")
            return

    # ── Done ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Training complete!")
    print(f"  Best Dice: {best_val_dice:.4f}")
    print(f"{'='*70}\n")

    config_path = os.path.join(CONFIG["output_dir"], "config.json")
    with open(config_path, "w") as f:
        # Patch tuple to list for JSON serialization
        cfg = {**CONFIG, "patch_size": list(CONFIG["patch_size"])}
        json.dump(cfg, f, indent=2)
    print(f"  Config  → {config_path}")
    print(f"  History → {os.path.join(CONFIG['output_dir'], 'history.json')}")


if __name__ == "__main__":
    train()
