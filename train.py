#!/usr/bin/env python3
"""
Main training script with GPU/MPS optimization.
Loads config, trains model, saves results.
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from utils.model import TumorCNN
from utils.dataset import get_dataloaders
from utils.training_utils import (
    get_device, get_optimizer_and_scheduler, setup_amp,
    train_one_epoch, evaluate
)


def train(config_path, output_dir="models"):
    """Train the model using config."""
    with open(config_path) as f:
        config = json.load(f)

    # Setup
    device = get_device(config["device"])
    config["device"] = device  # Update config with actual device
    os.makedirs(output_dir, exist_ok=True)

    # Data
    train_loader, test_loader, classes = get_dataloaders(config)
    print(f"Classes: {classes}")
    print(f"Train: {len(train_loader.dataset)} images | Test: {len(test_loader.dataset)} images")

    # Model
    model = TumorCNN(config).to(device)
    print(f"Model parameters: {model.get_total_params():,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, config)
    use_amp, scaler = setup_amp(device)

    if device == "cuda" and use_amp:
        print("✓ Mixed Precision (AMP) enabled - 1.5-2x faster training")

    # Training loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    best_model_path = os.path.join(output_dir, "best_model.pth")

    epochs = config["training"]["epochs"]
    print(f"\n{'='*70}")
    print(f"{'Training':<35} {'Batch Size: ' + str(config['training']['batch_size']):<35}")
    print(f"{'='*70}\n")

    epoch_times = []
    epoch_pbar = tqdm(range(1, epochs + 1), desc="Training Progress", unit="epoch")

    for epoch in epoch_pbar:
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler,
            epoch=epoch, total_epochs=epochs
        )
        val_loss, val_acc, _, _ = evaluate(
            model, test_loader, criterion, device, use_amp,
            epoch=epoch, total_epochs=epochs, stage="Val"
        )

        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch
        eta_seconds = remaining_epochs * avg_time
        eta_mins = eta_seconds / 60

        mem_str = ""
        if device == "cuda":
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_str = f" | GPU: {mem_used:.2f}GB"

        status = f"Loss: {train_loss:.4f}→{val_loss:.4f} | Acc: {train_acc:.3f}→{val_acc:.3f} | {elapsed:.1f}s | ETA: {eta_mins:.1f}m{mem_str}"
        epoch_pbar.set_postfix_str(status)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            epoch_pbar.write(f"        ✓ Best model saved (val_acc={best_acc:.3f})")

    # Save training history
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save config used
    config_out_path = os.path.join(output_dir, "config.json")
    with open(config_out_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best val accuracy: {best_acc:.3f}")
    print(f"Model saved to: {best_model_path}")
    print(f"History saved to: {history_path}")
    print(f"{'='*70}\n")

    return best_model_path, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train brain tumor classifier")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--output", default="models", help="Output directory for weights")
    args = parser.parse_args()

    train(args.config, args.output)
