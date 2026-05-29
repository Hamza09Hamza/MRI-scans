#!/usr/bin/env python3
"""
Evaluate a trained model on test set.
Loads weights and config, prints detailed metrics.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from utils.model import TumorCNN
from utils.dataset import get_dataloaders
from utils.training_utils import get_device, evaluate


def load_model(config, weights_path, device):
    """Load model with saved weights."""
    model = TumorCNN(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


def evaluate_model(config_path, weights_path, output_dir=None):
    """Evaluate model and save metrics."""
    # Load config and weights
    with open(config_path) as f:
        config = json.load(f)

    device = get_device(config.get("device", "auto"))
    config["device"] = device

    # Create output directory
    if output_dir is None:
        output_dir = os.path.dirname(weights_path)
    os.makedirs(output_dir, exist_ok=True)

    # Data
    _, test_loader, classes = get_dataloaders(config)

    # Model
    model = load_model(config, weights_path, device)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*70}")
    print(f"Evaluating model: {weights_path}")
    print(f"Config: {config_path}")
    print(f"{'='*70}\n")

    # Evaluate
    use_amp = device in ["cuda", "mps"]
    val_loss, val_acc, preds, labels_true = evaluate(model, test_loader, criterion, device, use_amp)

    print(f"Test Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {val_acc:.3f}\n")

    # Classification report
    print("Per-class metrics:")
    print("-" * 70)
    report = classification_report(labels_true, preds, target_names=classes, digits=4)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(labels_true, preds)
    print("Confusion Matrix:")
    print("-" * 70)
    print(cm)
    print()

    # Save metrics
    metrics = {
        "test_loss": float(val_loss),
        "test_accuracy": float(val_acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classes": classes,
    }

    metrics_path = os.path.join(output_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix plot saved to: {cm_path}")
    print(f"{'='*70}\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--config", default="config.json", help="Config file path")
    parser.add_argument("--output", help="Output directory for metrics (default: same as weights)")
    args = parser.parse_args()

    evaluate_model(args.config, args.weights, args.output)
