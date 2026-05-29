#!/usr/bin/env python3
"""
Test the combined BraTS2020 + UPENN-GBM dataset loader.
"""

from utils.multi_dataset_3d import create_combined_dataset

print("Testing combined dataset loader...\n")

# Test BraTS2020 training split
print("=" * 70)
print("BraTS2020 (training)")
print("=" * 70)
try:
    brats_train = create_combined_dataset(
        brats_dir="./data/BraTS2020",
        upenn_dir=None,
        split="train",
        modalities=["t1ce", "flair"],
        patch_size=(64, 128, 128),
    )
    print(f"✓ BraTS2020 training: {len(brats_train)} volumes\n")

    if len(brats_train) > 0:
        sample = brats_train[0]
        print(f"  Sample from {sample['patient_id']} ({sample['dataset']})")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Mask classes: {set(sample['mask'].unique().tolist())}\n")
except Exception as e:
    print(f"✗ BraTS2020 error: {e}\n")

# Test BraTS2020 validation split
print("=" * 70)
print("BraTS2020 (validation)")
print("=" * 70)
try:
    brats_val = create_combined_dataset(
        brats_dir="./data/BraTS2020",
        upenn_dir=None,
        split="val",
        modalities=["t1ce", "flair"],
        patch_size=(64, 128, 128),
    )
    print(f"✓ BraTS2020 validation: {len(brats_val)} volumes\n")
except Exception as e:
    print(f"✗ BraTS2020 validation error: {e}\n")

# Test UPENN-GBM
print("=" * 70)
print("UPENN-GBM")
print("=" * 70)
try:
    upenn = create_combined_dataset(
        brats_dir=None,
        upenn_dir="./data/upenn_gbm_nifti",
        modalities=["t1ce", "flair"],
        patch_size=(64, 128, 128),
    )
    print(f"✓ UPENN-GBM: {len(upenn)} volumes\n")

    if len(upenn) > 0:
        sample = upenn[0]
        print(f"  Sample from {sample['patient_id']} ({sample['dataset']})")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Mask classes: {set(sample['mask'].unique().tolist())}\n")
except Exception as e:
    print(f"✗ UPENN-GBM error: {e}\n")

# Test combined dataset
print("=" * 70)
print("Combined Dataset (BraTS2020 + UPENN-GBM)")
print("=" * 70)
try:
    combined = create_combined_dataset(
        brats_dir="./data/BraTS2020",
        upenn_dir="./data/upenn_gbm_nifti",
        split="train",
        modalities=["t1ce", "flair"],
        patch_size=(64, 128, 128),
    )
    print(f"✓ Combined dataset: {len(combined)} total volumes\n")
except Exception as e:
    print(f"✗ Combined error: {e}\n")

print("=" * 70)
print("Test complete!")
print("=" * 70)
