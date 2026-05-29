#!/usr/bin/env python3
"""
Load pre-trained 3D U-Net model for brain tumor segmentation.

Usage:
    from SEGMENTATION.load_model import load_segmentation_model

    model, device = load_segmentation_model()
    output = model(input_tensor)  # (batch, 4, 64, 128, 128)
"""

import os
import torch
import torch.nn as nn
from pathlib import Path


def load_segmentation_model(weights_path=None, device=None):
    """
    Load pre-trained 3D U-Net for brain tumor segmentation.

    Args:
        weights_path: Path to model weights (.pth file)
                     Default: SEGMENTATION/weights/unet3d_segmentation_best.pth
        device: torch device ('cuda', 'mps', 'cpu')
               Default: auto-detect (cuda > mps > cpu)

    Returns:
        model: UNet3D model with loaded weights
        device: torch device used

    Example:
        >>> model, device = load_segmentation_model()
        >>> # Input: (batch, 2, 64, 128, 128)
        >>> # Output: (batch, 4, 64, 128, 128)
        >>> output = model(input_tensor)
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
        weights_path = Path(__file__).parent / "weights" / "unet3d_segmentation_best.pth"

    weights_path = Path(weights_path)

    # Validate path
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}\n"
            f"Download from: https://github.com/Hamza09Hamza/MRI-scans/releases\n"
            f"Or train your own: python3 train_3d_segmentation.py"
        )

    # Import UNet3D
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.unet_3d import UNet3D

    # Create model
    model = UNet3D(in_channels=2, out_channels=4, features=32)

    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)

    # Handle DataParallel wrapper
    if isinstance(checkpoint, dict) and "module." in list(checkpoint.keys())[0]:
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print(f"✓ 3D U-Net Segmentation Model Loaded")
    print(f"  Weights: {weights_path}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Input shape: (batch, 2, 64, 128, 128)")
    print(f"  Output shape: (batch, 4, 64, 128, 128)")
    print(f"  Status: Evaluation mode (inference ready)\n")

    return model, device


def get_model_info():
    """Return model architecture and configuration."""
    return {
        "architecture": "3D U-Net",
        "input_channels": 2,
        "output_channels": 4,
        "input_shape": (1, 2, 64, 128, 128),
        "output_shape": (1, 4, 64, 128, 128),
        "patch_size": (64, 128, 128),
        "parameters": 22_582_180,
        "training_data": "BraTS2020 (369) + UPENN-GBM (5)",
        "total_volumes": 374,
        "modalities": ["T1ce (contrast-enhanced T1)", "FLAIR (fluid-suppressed T2)"],
        "classes": {
            0: "Background",
            1: "Necrotic Core",
            2: "Edema",
            3: "Enhancing Tumor",
        },
        "expected_dice": "0.65-0.75+",
        "inference_speed_gpu": "~1.2s per volume",
        "inference_speed_cpu": "~10s per volume",
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  3D Brain Tumor Segmentation Model")
    print("=" * 70 + "\n")

    # Print model info
    info = get_model_info()
    print("Model Specifications:")
    print(f"  Architecture: {info['architecture']}")
    print(f"  Input: {info['input_channels']} channels (T1ce + FLAIR)")
    print(f"  Output: {info['output_channels']} classes")
    print(f"  Parameters: {info['parameters']:,}")
    print(f"  Patch size: {info['patch_size']}")
    print(f"  Expected Dice: {info['expected_dice']}")
    print()

    # Load model
    print("Loading model...")
    try:
        model, device = load_segmentation_model()
        print("✓ Model ready for inference!")
        print("\nUsage:")
        print("  from SEGMENTATION.load_model import load_segmentation_model")
        print("  model, device = load_segmentation_model()")
        print("  output = model(input_tensor)  # input: (B, 2, 64, 128, 128)")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
