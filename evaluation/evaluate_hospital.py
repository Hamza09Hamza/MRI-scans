#!/usr/bin/env python3
"""
Evaluate trained model on the Epic & CSCR hospital dataset.
This is a real-world generalization test — model was NOT trained on this data.

Usage:
    python evaluate_hospital.py
    python evaluate_hospital.py --weights models/best_model.pth --config config.json
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.model import TumorCNN
from utils.training_utils import get_device

DATASET_DIR = "Epic and CSCR hospital Dataset/Test"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def evaluate_hospital(weights_path, config_path, output_dir="hospital_eval"):
    with open(config_path) as f:
        config = json.load(f)

    device = get_device(config.get("device", "auto"), verbose=False)
    config["device"] = device
    img_size = config["data"]["img_size"]

    # Load model
    model = TumorCNN(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Load hospital dataset
    dataset = datasets.ImageFolder(DATASET_DIR, transform=get_transform(img_size))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    classes = dataset.classes
    total_images = len(dataset)

    print(f"\n{'='*70}")
    print(f"Evaluating on: {DATASET_DIR}")
    print(f"Total images : {total_images}")
    print(f"Classes      : {classes}")
    print(f"Model weights: {weights_path}")
    print(f"{'='*70}\n")

    # Run inference
    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            if device in ["cuda", "mps"]:
                with torch.amp.autocast(device_type=device):
                    outputs = model(images)
            else:
                outputs = model(images)

            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"acc": f"{correct/total:.3f}"})

    # Results
    overall_acc = correct / total

    print(f"\n{'='*70}")
    print(f"RESULTS ON EPIC & CSCR HOSPITAL DATASET")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {overall_acc*100:.2f}%  ({correct}/{total})\n")

    print("Per-class breakdown:")
    print("-" * 70)
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print("-" * 70)
    print(cm)

    # Per-class accuracy summary
    print("\nPer-class accuracy:")
    for i, cls in enumerate(classes):
        cls_total = cm[i].sum()
        cls_correct = cm[i][i]
        print(f"  {cls:<12}: {cls_correct}/{cls_total}  ({100*cls_correct/cls_total:.1f}%)")

    # Save confusion matrix plot
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Hospital Dataset — Overall Accuracy: {overall_acc*100:.2f}%")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # Save metrics JSON
    metrics = {
        "dataset": DATASET_DIR,
        "model_weights": weights_path,
        "total_images": total_images,
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": {
            cls: float(cm[i][i] / cm[i].sum())
            for i, cls in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nConfusion matrix saved to: {cm_path}")
    print(f"Metrics saved to         : {metrics_path}")
    print(f"{'='*70}\n")

    return overall_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on hospital dataset")
    parser.add_argument("--weights", default="models/best_model.pth")
    parser.add_argument("--config",  default="config.json")
    parser.add_argument("--output",  default="hospital_eval")
    args = parser.parse_args()

    evaluate_hospital(args.weights, args.config, args.output)
