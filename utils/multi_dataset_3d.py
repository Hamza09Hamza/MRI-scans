"""
Multi-dataset 3D loader for brain tumor segmentation.
Supports both UPENN-GBM and BraTS2020 datasets with unified interface.
"""

import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


class BraTS2020Dataset(Dataset):
    """
    Load BraTS2020 3D volumes (already in NIfTI format).

    Structure:
        BraTS2020/
        ├── BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
        │   ├── BraTS20_Training_001/
        │   │   ├── BraTS20_Training_001_t1.nii.gz
        │   │   ├── BraTS20_Training_001_t1ce.nii.gz
        │   │   ├── BraTS20_Training_001_t2.nii.gz
        │   │   ├── BraTS20_Training_001_flair.nii.gz
        │   │   └── BraTS20_Training_001_seg.nii.gz
        │   ├── BraTS20_Training_002/
        ...
    """

    def __init__(self, data_dir, split="train", modalities=["t1ce", "flair"], patch_size=(64, 128, 128)):
        """
        Args:
            data_dir: Path to BraTS2020 root directory
            split: "train" or "val" (validation)
            modalities: List of modality names (t1, t1ce, t2, flair)
            patch_size: (depth, height, width)
        """
        self.data_dir = data_dir
        self.modalities = modalities
        self.patch_size = patch_size
        self.scaler = MinMaxScaler()

        # Find the correct data directory
        if split == "train":
            base_path = os.path.join(data_dir, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
        else:
            base_path = os.path.join(data_dir, "BraTS2020_ValidationData", "MICCAI_BraTS2020_ValidationData")

        # Find all patient directories
        all_patient_dirs = sorted(glob.glob(os.path.join(base_path, "BraTS20_*")))

        if not all_patient_dirs:
            raise FileNotFoundError(f"No BraTS2020 patients found in {base_path}")

        # Filter to only include patients with segmentation files
        self.patient_dirs = []
        for pdir in all_patient_dirs:
            patient_id = os.path.basename(pdir)
            seg_pattern = os.path.join(pdir, f"{patient_id}_seg*.nii*")
            if glob.glob(seg_pattern):  # Only include if segmentation exists
                self.patient_dirs.append(pdir)

        print(f"BraTS2020 ({split}): Found {len(self.patient_dirs)} patients with segmentation (out of {len(all_patient_dirs)} total)")

    def __len__(self):
        return len(self.patient_dirs)

    def load_nifti(self, path):
        """Load NIfTI file"""
        if not os.path.exists(path):
            return None
        img = nib.load(path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        # Remove extra dimensions with size 1 (e.g., time dimension)
        data = np.squeeze(data)
        # Ensure 3D
        if data.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {data.shape}")
        return data

    def normalize_volume(self, volume):
        """Normalize to [0, 1]"""
        v_min, v_max = volume.min(), volume.max()
        if v_max - v_min > 0:
            return (volume - v_min) / (v_max - v_min)
        return volume

    def crop_center(self, volume, target_shape):
        """Crop from center"""
        current_shape = np.array(volume.shape)
        target_shape = np.array(target_shape)

        if np.all(current_shape <= target_shape):
            return volume

        start_idx = (current_shape - target_shape) // 2
        end_idx = start_idx + target_shape

        slices = tuple(slice(start_idx[i], end_idx[i]) for i in range(len(target_shape)))
        return volume[slices]

    def pad_volume(self, volume, target_shape):
        """Pad to target shape"""
        current_shape = np.array(volume.shape)
        target_shape = np.array(target_shape)

        # Handle each dimension: pad if too small, crop if too large
        padding = []
        for i in range(len(target_shape)):
            if current_shape[i] < target_shape[i]:
                # Pad this dimension
                pad_amount = target_shape[i] - current_shape[i]
                padding.append((pad_amount // 2, pad_amount - pad_amount // 2))
            else:
                # No padding needed for this dimension
                padding.append((0, 0))

        # Apply padding
        volume = np.pad(volume, padding, mode='constant', constant_values=0)

        # Now crop if any dimension is still too large
        if np.any(np.array(volume.shape) > target_shape):
            volume = self.crop_center(volume, target_shape)

        return volume

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id = os.path.basename(patient_dir)

        # Load modalities
        modality_volumes = []
        for modality in self.modalities:
            # BraTS naming: BraTS20_Training_001_t1ce.nii.gz
            pattern = os.path.join(patient_dir, f"{patient_id}_{modality}*.nii*")
            matches = glob.glob(pattern)

            if not matches:
                raise FileNotFoundError(f"Could not find {modality} for {patient_id}")

            volume = self.load_nifti(matches[0])
            volume = self.normalize_volume(volume)
            volume = self.pad_volume(volume, self.patch_size)
            modality_volumes.append(volume)

        # Stack modalities
        x = np.stack(modality_volumes, axis=0).astype(np.float32)

        # Load segmentation
        seg_pattern = os.path.join(patient_dir, f"{patient_id}_seg*.nii*")
        seg_matches = glob.glob(seg_pattern)

        if not seg_matches:
            raise FileNotFoundError(f"Could not find segmentation for {patient_id}")

        y = self.load_nifti(seg_matches[0])
        y = self.pad_volume(y, self.patch_size).astype(np.int64)

        # Fix: map label 4 → 3
        y[y == 4] = 3

        return {
            "image": torch.from_numpy(x),
            "mask": torch.from_numpy(y),
            "patient_id": patient_id,
            "dataset": "BraTS2020",
        }


class UPENNDBM_NIfTIDataset(Dataset):
    """
    Load UPENN-GBM after DICOM→NIfTI conversion.

    Structure (after conversion):
        upenn_gbm_nifti/
        ├── UPENN-GBM-00001/
        │   ├── T1ce.nii.gz
        │   ├── FLAIR.nii.gz
        │   └── segmentation.nii.gz
        ├── UPENN-GBM-00002/
        ...
    """

    def __init__(self, data_dir, modalities=["T1ce", "FLAIR"], patch_size=(64, 128, 128)):
        self.data_dir = data_dir
        self.modalities = modalities
        self.patch_size = patch_size
        self.scaler = MinMaxScaler()

        self.patient_dirs = sorted(glob.glob(os.path.join(data_dir, "UPENN-GBM-*")))

        if not self.patient_dirs:
            raise FileNotFoundError(f"No UPENN-GBM patients found in {data_dir}")

        print(f"UPENN-GBM: Found {len(self.patient_dirs)} patients")

    def __len__(self):
        return len(self.patient_dirs)

    def load_nifti(self, path):
        if not os.path.exists(path):
            return None
        img = nib.load(path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        # Remove extra dimensions with size 1 (e.g., time dimension)
        data = np.squeeze(data)
        # Ensure 3D
        if data.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {data.shape}")
        return data

    def normalize_volume(self, volume):
        v_min, v_max = volume.min(), volume.max()
        if v_max - v_min > 0:
            return (volume - v_min) / (v_max - v_min)
        return volume

    def crop_center(self, volume, target_shape):
        current_shape = np.array(volume.shape)
        target_shape = np.array(target_shape)
        if np.all(current_shape <= target_shape):
            return volume
        start_idx = (current_shape - target_shape) // 2
        end_idx = start_idx + target_shape
        slices = tuple(slice(start_idx[i], end_idx[i]) for i in range(len(target_shape)))
        return volume[slices]

    def pad_volume(self, volume, target_shape):
        current_shape = np.array(volume.shape)
        target_shape = np.array(target_shape)

        # Handle each dimension: pad if too small, crop if too large
        padding = []
        for i in range(len(target_shape)):
            if current_shape[i] < target_shape[i]:
                # Pad this dimension
                pad_amount = target_shape[i] - current_shape[i]
                padding.append((pad_amount // 2, pad_amount - pad_amount // 2))
            else:
                # No padding needed for this dimension
                padding.append((0, 0))

        # Apply padding
        volume = np.pad(volume, padding, mode='constant', constant_values=0)

        # Now crop if any dimension is still too large
        if np.any(np.array(volume.shape) > target_shape):
            volume = self.crop_center(volume, target_shape)

        return volume

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id = os.path.basename(patient_dir)

        modality_volumes = []
        for modality in self.modalities:
            candidates = [
                os.path.join(patient_dir, f"{modality}.nii.gz"),
                os.path.join(patient_dir, f"{modality}.nii"),
            ]

            volume = None
            for candidate in candidates:
                if os.path.exists(candidate):
                    volume = self.load_nifti(candidate)
                    break

            if volume is None:
                raise FileNotFoundError(f"Could not find {modality} for {patient_id}")

            volume = self.normalize_volume(volume)
            volume = self.pad_volume(volume, self.patch_size)
            modality_volumes.append(volume)

        x = np.stack(modality_volumes, axis=0).astype(np.float32)

        seg_path = os.path.join(patient_dir, "segmentation.nii.gz")
        if not os.path.exists(seg_path):
            seg_path = os.path.join(patient_dir, "segmentation.nii")

        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Could not find segmentation for {patient_id}")

        y = self.load_nifti(seg_path)
        y = self.pad_volume(y, self.patch_size).astype(np.int64)
        y[y == 4] = 3

        return {
            "image": torch.from_numpy(x),
            "mask": torch.from_numpy(y),
            "patient_id": patient_id,
            "dataset": "UPENN-GBM",
        }


def create_combined_dataset(brats_dir=None, upenn_dir=None, split="train",
                           modalities=["t1ce", "flair"], patch_size=(64, 128, 128)):
    """
    Create a combined dataset from BraTS2020 and UPENN-GBM.

    Args:
        brats_dir: Path to BraTS2020 root directory
        upenn_dir: Path to UPENN-GBM NIfTI directory (after conversion)
        split: "train" or "val"
        modalities: List of modality names (lowercase for BraTS2020: ["t1ce", "flair"])
        patch_size: Patch size

    Returns:
        Combined torch ConcatDataset
    """
    datasets = []

    if brats_dir and os.path.exists(brats_dir):
        print(f"\nLoading BraTS2020 ({split} split)...")
        brats_dataset = BraTS2020Dataset(
            brats_dir,
            split=split,
            modalities=modalities,
            patch_size=patch_size
        )
        datasets.append(brats_dataset)

    if upenn_dir and os.path.exists(upenn_dir):
        print(f"Loading UPENN-GBM...")
        # Convert modalities to UPENN naming convention (capitalize first letter)
        upenn_modalities = [m.title() if m.lower() in ["t1ce", "flair", "t1", "t2"] else m
                           for m in modalities]
        upenn_dataset = UPENNDBM_NIfTIDataset(
            upenn_dir,
            modalities=upenn_modalities,
            patch_size=patch_size
        )
        datasets.append(upenn_dataset)

    if not datasets:
        raise ValueError("No datasets found! Provide valid brats_dir and/or upenn_dir")

    if len(datasets) == 1:
        return datasets[0]
    else:
        total_size = sum(len(d) for d in datasets)
        print(f"\nCombined dataset: {total_size} volumes")
        return ConcatDataset(datasets)


if __name__ == "__main__":
    # Test loading both datasets
    brats_dir = "./data/BraTS2020"
    upenn_dir = "./data/upenn_gbm_nifti"

    print("Testing BraTS2020...")
    try:
        brats_train = BraTS2020Dataset(brats_dir, split="train", modalities=["t1ce", "flair"])
        sample = brats_train[0]
        print(f"✓ BraTS2020 sample: {sample['patient_id']}")
        print(f"  Image: {sample['image'].shape}, Mask: {sample['mask'].shape}")
    except Exception as e:
        print(f"✗ BraTS2020 error: {e}")

    print("\nTesting UPENN-GBM...")
    try:
        upenn = UPENNDBM_NIfTIDataset(upenn_dir, modalities=["T1ce", "FLAIR"])
        sample = upenn[0]
        print(f"✓ UPENN-GBM sample: {sample['patient_id']}")
        print(f"  Image: {sample['image'].shape}, Mask: {sample['mask'].shape}")
    except Exception as e:
        print(f"✗ UPENN-GBM error: {e}")

    print("\nTesting combined dataset...")
    try:
        combined = create_combined_dataset(brats_dir, upenn_dir)
        print(f"✓ Combined dataset size: {len(combined)}")
    except Exception as e:
        print(f"✗ Combined error: {e}")
