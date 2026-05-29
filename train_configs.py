"""
Pre-configured training setups for different hardware.
Edit and copy the appropriate config into train_3d_segmentation.py
"""

# M1 Pro / M2 Pro / M3 Pro (16GB+ Unified Memory)
CONFIG_M1_PRO = {
    "brats_dir": "./data/BraTS2020",
    "upenn_dir": "./data/upenn_gbm_nifti",
    "output_dir": "./segmentation_models",
    "model_name": "unet3d_multimodal",
    "device": "mps",  # Automatic MPS if available
    "epochs": 50,
    "batch_size": 2,  # Safe for 16GB unified memory
    "learning_rate": 1e-4,
    "num_workers": 2,  # Can use more workers on M1
    "pin_memory": False,
    "modalities": ["t1ce", "flair"],
    "patch_size": (64, 128, 128),
    "num_classes": 4,
    "in_channels": 2,
}

# RTX 3050 (6GB VRAM, 24GB+ RAM) - Conservative (slower training)
CONFIG_RTX_3050_SAFE = {
    "brats_dir": "./data/BraTS2020",
    "upenn_dir": "./data/upenn_gbm_nifti",
    "output_dir": "./segmentation_models",
    "model_name": "unet3d_multimodal_safe",
    "device": "cuda",  # Force CUDA
    "epochs": 50,
    "batch_size": 1,  # Safe: minimal memory usage
    "learning_rate": 1e-4,
    "num_workers": 2,
    "pin_memory": True,  # Pin memory: use RAM as buffer for VRAM
    "modalities": ["t1ce", "flair"],
    "patch_size": (64, 128, 128),
    "num_classes": 4,
    "in_channels": 2,
    "use_gradient_checkpointing": False,
}

# RTX 3050 (6GB VRAM, 24GB+ RAM) - With Pinned Memory (RECOMMENDED)
CONFIG_RTX_3050 = {
    "brats_dir": "./data/BraTS2020",
    "upenn_dir": "./data/upenn_gbm_nifti",
    "output_dir": "./segmentation_models",
    "model_name": "unet3d_multimodal",
    "device": "cuda",
    "epochs": 50,
    "batch_size": 2,  # Now safe with pin_memory + larger RAM
    "learning_rate": 1e-4,
    "num_workers": 2,
    "pin_memory": True,  # KEY: Use 24GB RAM as additional VRAM buffer (slower but works)
    "modalities": ["t1ce", "flair"],
    "patch_size": (64, 128, 128),
    "num_classes": 4,
    "in_channels": 2,
    "use_gradient_checkpointing": False,
}

# RTX 3050 with Gradient Checkpointing (AGGRESSIVE - uses CPU RAM + GPU VRAM smartly)
CONFIG_RTX_3050_AGGRESSIVE = {
    "brats_dir": "./data/BraTS2020",
    "upenn_dir": "./data/upenn_gbm_nifti",
    "output_dir": "./segmentation_models",
    "model_name": "unet3d_multimodal_aggressive",
    "device": "cuda",
    "epochs": 50,
    "batch_size": 3,  # Increased batch size with checkpointing + pinned memory
    "learning_rate": 1e-4,
    "num_workers": 2,
    "pin_memory": True,  # Uses 24GB system RAM as VRAM buffer
    "modalities": ["t1ce", "flair"],
    "patch_size": (64, 128, 128),
    "num_classes": 4,
    "in_channels": 2,
    "use_gradient_checkpointing": True,  # Recompute activations instead of storing (~30% memory savings)
}

# RTX 4090 / RTX 3090 (24GB VRAM) - Recommended
CONFIG_RTX_4090 = {
    "brats_dir": "./data/BraTS2020",
    "upenn_dir": "./data/upenn_gbm_nifti",
    "output_dir": "./segmentation_models",
    "model_name": "unet3d_multimodal",
    "device": "cuda",
    "epochs": 50,
    "batch_size": 4,  # Can use larger batch on 24GB
    "learning_rate": 1e-4,
    "num_workers": 4,
    "pin_memory": True,
    "modalities": ["t1ce", "flair"],
    "patch_size": (64, 128, 128),
    "num_classes": 4,
    "in_channels": 2,
}

# CPU Only (not recommended, very slow)
CONFIG_CPU = {
    "brats_dir": "./data/BraTS2020",
    "upenn_dir": "./data/upenn_gbm_nifti",
    "output_dir": "./segmentation_models",
    "model_name": "unet3d_multimodal_cpu",
    "device": "cpu",
    "epochs": 50,
    "batch_size": 1,
    "learning_rate": 1e-4,
    "num_workers": 0,  # Don't use workers on CPU
    "pin_memory": False,
    "modalities": ["t1ce", "flair"],
    "patch_size": (64, 128, 128),
    "num_classes": 4,
    "in_channels": 2,
}

# Tips for your hardware:
#
# M1 Pro 16GB Unified Memory:
#   - Use CONFIG_M1_PRO
#   - Can safely increase batch_size to 3-4 if you have more memory
#   - MPS training is typically 1.5-2x faster than CPU
#   - pin_memory=False is OK (unified memory is already fast)
#
# RTX 3050 6GB VRAM + 24GB RAM:
#   THREE OPTIONS (in order of preference):
#
#   1. RECOMMENDED: CONFIG_RTX_3050 (batch_size=2)
#      - pin_memory=True: Uses 24GB RAM as unified VRAM buffer
#      - Slower than pure VRAM but allows batch_size=2
#      - Training time: ~2.5-4 hours for 50 epochs
#      - ~5-10% slower than batch_size=1 on pure VRAM
#
#   2. BALANCED: CONFIG_RTX_3050_AGGRESSIVE (batch_size=3)
#      - Combines gradient checkpointing + pinned memory
#      - Trades compute for memory (recomputes gradients)
#      - ~25% slower per batch, but larger batches improve convergence
#      - Training time: ~2-3.5 hours (faster overall due to larger batches)
#      - Best accuracy/time tradeoff
#
#   3. SAFE: CONFIG_RTX_3050_SAFE (batch_size=1)
#      - Pure VRAM usage, no RAM borrowing
#      - Slowest per epoch but very stable
#      - Use if experiencing crashes with configs 1-2
#
# How Pinned Memory Works:
#   - pin_memory=True allocates 24GB system RAM as "pinned" (fast CPU↔GPU transfers)
#   - GPU can directly access pinned RAM over PCIe (slower than VRAM but much faster than normal RAM)
#   - Allows batch_size=2 on 6GB VRAM by spilling overflow to pinned RAM
#   - Speedup vs batch_size=1: ~30-40% fewer iterations, worth the PCIe bandwidth cost
#
# Gradient Checkpointing:
#   - Trades 25-30% more compute for 30% memory savings
#   - Recomputes activations during backward pass instead of storing them
#   - Only enable if you hit OOM errors with pinned memory
#   - Slows each batch by ~15% but allows larger batches
#
# RTX 4090 / 3090 24GB VRAM:
#   - Use CONFIG_RTX_4090
#   - Can safely go up to batch_size=8 with this model
#   - Consider mixed precision training (torch.cuda.amp) for faster training
#   - pin_memory=True still helps (GPU can prefetch while computing)
#
# CPU Only:
#   - Only for debugging, not practical for full training
#   - One epoch will take 30+ minutes
#   - Use for quick dataset testing instead
