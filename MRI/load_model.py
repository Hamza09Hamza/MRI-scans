#!/usr/bin/env python3
"""
Load pre-trained ResNet50 brain tumor classifier.

Usage:
    from MRI.load_model import load_brain_classifier

    model, device = load_brain_classifier()
    # model ready for inference
"""

import os
import torch
import torch.nn as nn
from pathlib import Path


def load_brain_classifier(weights_path=None, device=None):
    """
    Load pre-trained ResNet50 brain tumor classifier.

    Args:
        weights_path: Path to model weights (.pth file)
                     Default: MRI/weights/brain_tumor_classifier_best.pth
        device: torch device ('cuda', 'mps', 'cpu')
               Default: auto-detect (cuda > mps > cpu)

    Returns:
        model: ResNet50 classifier with loaded weights
        device: torch device used

    Example:
        >>> model, device = load_brain_classifier()
        >>> # Now model is ready for inference
        >>> predictions = model(mri_tensor)
    """

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Default weights path
    if weights_path is None:
        weights_path = Path(__file__).parent / "weights" / "brain_tumor_classifier_best.pth"

    weights_path = Path(weights_path)

    # Validate path
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}\n"
            f"Download from: https://github.com/Hamza09Hamza/MRI-scans/releases\n"
            f"Or train your own: python3 train.py"
        )

    # Create model: ResNet50 backbone + clinical classifier head
    from torchvision import models

    # Build model matching the architecture used in training
    class BrainTumorClassifier(nn.Module):
        def __init__(self, num_classes=4, dropout=0.3):
            super().__init__()
            # ResNet50 backbone
            self.backbone = models.resnet50(weights=None)  # Don't download
            num_features = self.backbone.fc.in_features  # 2048
            self.backbone.fc = nn.Identity()

            # Clinical classifier head
            self.classifier = nn.Sequential(
                nn.Linear(num_features, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)

    model = BrainTumorClassifier(num_classes=4, dropout=0.3)

    # Load pre-trained weights
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print(f"✓ Brain Tumor Classifier loaded")
    print(f"  Weights: {weights_path}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Status: Evaluation mode (inference ready)\n")

    return model, device


def get_model_info():
    """Return model architecture and configuration."""
    return {
        "architecture": "ResNet50",
        "input_shape": (1, 3, 128, 128),
        "input_size": 128,
        "input_channels": 3,
        "output_channels": 4,
        "parameters": 26_000_000,
        "trainable_parameters": 1_000_000,
        "classes": {
            0: "Normal",
            1: "Pituitary",
            2: "Meningioma",
            3: "Glioma",
        },
        "training_data": "Kaggle Brain Tumor Dataset (~3,000 images)",
        "validation_data": "Epic & CSCR Hospital Dataset (1,857 images)",
        "accuracy": 0.9736,
        "accuracy_percentage": "97.36%",
        "inference_speed_gpu": "~10ms per image",
        "inference_speed_cpu": "~100ms per image",
    }


def preprocess_image(image_path, device):
    """
    Load and preprocess a single brain MRI image.

    Args:
        image_path: Path to image file (JPG, PNG, etc.)
        device: torch device

    Returns:
        tensor: (1, 3, 128, 128) preprocessed image tensor
    """
    from PIL import Image
    import torchvision.transforms as transforms

    # Load image
    img = Image.open(image_path).convert("RGB")

    # ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return transform(img).unsqueeze(0).to(device)


def classify_image(image_path, model, device):
    """
    Classify a brain MRI image.

    Args:
        image_path: Path to image file
        model: Loaded brain classifier model
        device: torch device

    Returns:
        dict with prediction, confidence, and all class probabilities
    """
    # Preprocess
    x = preprocess_image(image_path, device)

    # Predict
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    classes = ["Normal", "Pituitary", "Meningioma", "Glioma"]
    pred_idx = torch.argmax(probs, dim=1).item()

    return {
        "image": image_path,
        "prediction": classes[pred_idx],
        "confidence": float(probs[0, pred_idx].item()),
        "all_scores": {cls: float(prob) for cls, prob in zip(classes, probs[0])},
    }


if __name__ == "__main__":
    import json

    print("=" * 70)
    print("  Brain Tumor Classification Model")
    print("=" * 70 + "\n")

    # Print model info
    info = get_model_info()
    print("Model Specifications:")
    print(f"  Architecture: {info['architecture']}")
    print(f"  Input size: {info['input_size']}×{info['input_size']} RGB images")
    print(f"  Classes: {len(info['classes'])} (Normal, Pituitary, Meningioma, Glioma)")
    print(f"  Parameters: {info['parameters']:,}")
    print(f"  Accuracy: {info['accuracy_percentage']} (Hospital validation)")
    print(f"  Speed: {info['inference_speed_gpu']} (GPU)")
    print()

    # Load model
    print("Loading model...")
    try:
        model, device = load_brain_classifier()
        print("✓ Model ready for inference!")
        print("\nUsage:")
        print("  from MRI.load_model import load_brain_classifier, classify_image")
        print("  model, device = load_brain_classifier()")
        print("  result = classify_image('patient_mri.jpg', model, device)")
        print("  print(result['prediction'])")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
