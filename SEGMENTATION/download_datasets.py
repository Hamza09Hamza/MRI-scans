#!/usr/bin/env python3
"""
Download and setup datasets for 3D brain tumor segmentation.

Usage:
    python3 SEGMENTATION/download_datasets.py --brats       # BraTS2020 only
    python3 SEGMENTATION/download_datasets.py --full        # BraTS + UPENN-GBM
    python3 SEGMENTATION/download_datasets.py --check       # Verify downloaded data
"""

import os
import argparse
from pathlib import Path


def setup_data_directory():
    """Create data directory structure."""
    data_dirs = [
        "data",
        "data/BraTS2020",
        "data/upenn_gbm_nifti",
    ]
    for d in data_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Data directory structure created")


def download_brats2020():
    """Instructions for downloading BraTS2020."""
    print("\n" + "="*70)
    print("  BraTS2020 Download Instructions")
    print("="*70 + "\n")

    print("BraTS2020 is available on Synapse (requires free account)\n")

    print("Step 1: Create Synapse account")
    print("  - Go to: https://www.synapse.org/#!Synapse:syn22015602/files/")
    print("  - Sign up (free)\n")

    print("Step 2: Download datasets")
    print("  - Click 'Files' tab")
    print("  - Download:")
    print("    • BraTS2020_TrainingData/")
    print("    • BraTS2020_ValidationData/\n")

    print("Step 3: Extract to data/ folder")
    print("  unzip -q BraTS2020_TrainingData.zip -d data/BraTS2020/")
    print("  unzip -q BraTS2020_ValidationData.zip -d data/BraTS2020/\n")

    print("Step 4: Verify")
    print("  ls data/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/ | head")
    print("  # Should show: BraTS20_Training_001, BraTS20_Training_002, ...\n")

    print("Total size: ~50GB")
    print("Training volumes: 369")
    print("Validation volumes: 125")
    print("="*70 + "\n")


def download_upenn_gbm():
    """Instructions for downloading UPENN-GBM via IDC."""
    print("\n" + "="*70)
    print("  UPENN-GBM (DICOM) → NIfTI Conversion")
    print("="*70 + "\n")

    print("UPENN-GBM is available via IDC (Imaging Data Commons)\n")

    print("Option 1: Using IDC Index Package (Recommended)")
    print("  pip install idc-index")
    print("  python3 -c \"from idc.client import IDCClient; ...")
    print("  # Download UPENN-GBM DICOM files\n")

    print("Option 2: Batch download from IDC")
    print("  - Go to: https://imagingdatacommons.github.io/IDC-Free/")
    print("  - Search: 'UPENN-GBM'")
    print("  - Download all DICOM files\n")

    print("Step 2: Convert DICOM to NIfTI")
    print("  python3 utils/dicom_to_nifti.py")
    print("  # Converts to: data/upenn_gbm_nifti/\n")

    print("Step 3: Organize files")
    print("  python3 utils/organize_upenn_nifti.py")
    print("  # Creates standard naming: T1ce.nii.gz, FLAIR.nii.gz, segmentation.nii.gz\n")

    print("Total size: ~1GB (after DICOM→NIfTI conversion)")
    print("Volumes: 5 real clinical cases")
    print("="*70 + "\n")


def verify_datasets():
    """Verify downloaded datasets."""
    print("\n" + "="*70)
    print("  Dataset Verification")
    print("="*70 + "\n")

    # Check BraTS2020
    brats_path = Path("data/BraTS2020")
    print("Checking BraTS2020...")

    brats_train = brats_path / "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    brats_val = brats_path / "BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"

    if brats_train.exists():
        train_count = len(list(brats_train.glob("BraTS20_Training_*")))
        print(f"  ✓ Training data: {train_count} volumes")
    else:
        print(f"  ✗ Training data not found at {brats_train}")

    if brats_val.exists():
        val_count = len(list(brats_val.glob("BraTS20_Validation_*")))
        print(f"  ✓ Validation data: {val_count} volumes")
    else:
        print(f"  ✗ Validation data not found at {brats_val}")

    # Check UPENN-GBM
    print("\nChecking UPENN-GBM...")
    upenn_path = Path("data/upenn_gbm_nifti")

    if upenn_path.exists():
        upenn_count = len(list(upenn_path.glob("UPENN-GBM-*")))
        print(f"  ✓ UPENN-GBM: {upenn_count} volumes")
    else:
        print(f"  ✗ UPENN-GBM not found at {upenn_path}")

    print("\n" + "="*70)
    print("  Run test to verify data loads correctly:")
    print("  python3 test_combined_dataset.py")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download and setup datasets for 3D brain tumor segmentation"
    )
    parser.add_argument(
        "--brats",
        action="store_true",
        help="Setup for BraTS2020 only (recommended for testing)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Setup for full pipeline (BraTS2020 + UPENN-GBM)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify downloaded datasets"
    )

    args = parser.parse_args()

    # Setup directory structure
    setup_data_directory()

    if args.check:
        verify_datasets()
    elif args.brats or args.full:
        download_brats2020()
        if args.full:
            download_upenn_gbm()
    else:
        print("\n" + "="*70)
        print("  3D Brain Tumor Segmentation - Dataset Setup")
        print("="*70 + "\n")

        print("Usage:")
        print("  python3 SEGMENTATION/download_datasets.py --brats")
        print("    • Download BraTS2020 only (~50GB)")
        print("    • Recommended for testing\n")

        print("  python3 SEGMENTATION/download_datasets.py --full")
        print("    • Download BraTS2020 + UPENN-GBM")
        print("    • Complete pipeline (~51GB)\n")

        print("  python3 SEGMENTATION/download_datasets.py --check")
        print("    • Verify downloaded datasets\n")

        print("After downloading, run:")
        print("  python3 test_combined_dataset.py        # Verify data loads")
        print("  python3 train_3d_segmentation.py        # Start training")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
