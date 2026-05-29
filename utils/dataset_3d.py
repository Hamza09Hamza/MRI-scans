"""
3D Dataset loader for volumetric brain tumor segmentation.
Loads multi-modal NIfTI volumes with segmentation masks.
"""

import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class BrainTumorVolume3D(Dataset):
    """
    Load 3D brain tumor volumes with multi-modal MRI data.

    Expects directory structure:
        data_dir/
        ├── patient_001/
        │   ├── T1ce.nii.gz
        │   ├── FLAIR.nii.gz
        │   └── segmentation.nii.gz
        ├── patient_002/
        ...
    """

    def __init__(self, data_dir, modalities=["T1ce", "FLAIR"], patch_size=(64, 128, 128)):
        """
        Args:
            data_dir: Root directory containing patient folders
            modalities: List of modality names to load
            patch_size: Size of 3D patches (depth, height, width)
        """
        self.data_dir = data_dir
        self.modalities = modalities
        self.patch_size = patch_size
        self.scaler = MinMaxScaler()

        # Find all patient directories
        self.patient_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
        self.patient_dirs = [d for d in self.patient_dirs if os.path.isdir(d)]

        print(f"Found {len(self.patient_dirs)} patients in {data_dir}")

    def __len__(self):
        return len(self.patient_dirs)

    def load_nifti(self, path):
        """Load NIfTI file and return numpy array"""
        if not os.path.exists(path):
            return None
        img = nib.load(path)
        return np.asarray(img.dataobj, dtype=np.float32)

    def normalize_volume(self, volume):
        """Normalize volume to [0, 1] range"""
        volume = volume.reshape(-1, 1)
        normalized = self.scaler.fit_transform(volume)
        return normalized.reshape(volume.shape[:-1])

    def crop_center(self, volume, target_shape):
        """Crop volume to target shape from center"""
        current_shape = np.array(volume.shape)
        target_shape = np.array(target_shape)

        if np.all(current_shape <= target_shape):
            return volume  # No cropping needed

        start_idx = (current_shape - target_shape) // 2
        end_idx = start_idx + target_shape

        slices = tuple(slice(start_idx[i], end_idx[i]) for i in range(len(target_shape)))
        return volume[slices]

    def pad_volume(self, volume, target_shape):
        """Pad volume to target shape"""
        current_shape = np.array(volume.shape)
        target_shape = np.array(target_shape)

        if np.all(current_shape >= target_shape):
            return self.crop_center(volume, target_shape)

        pad_amount = target_shape - current_shape
        padding = tuple((pad_amount[i] // 2, pad_amount[i] - pad_amount[i] // 2) for i in range(len(target_shape)))
        return np.pad(volume, padding, mode='constant', constant_values=0)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id = os.path.basename(patient_dir)

        # Load modalities
        modality_volumes = []
        for modality in self.modalities:
            # Try different naming conventions
            candidates = [
                os.path.join(patient_dir, f"{modality}.nii.gz"),
                os.path.join(patient_dir, f"{modality}.nii"),
                os.path.join(patient_dir, f"*{modality}*.nii.gz"),
            ]

            volume = None
            for candidate in candidates:
                if "*" in candidate:
                    matches = glob.glob(candidate)
                    if matches:
                        volume = self.load_nifti(matches[0])
                        break
                elif os.path.exists(candidate):
                    volume = self.load_nifti(candidate)
                    break

            if volume is None:
                raise FileNotFoundError(f"Could not find {modality} for {patient_id}")

            # Normalize
            volume = self.normalize_volume(volume)

            # Resize to standard size
            volume = self.pad_volume(volume, self.patch_size)

            modality_volumes.append(volume)

        # Stack modalities (channels, depth, height, width)
        x = np.stack(modality_volumes, axis=0).astype(np.float32)

        # Load segmentation mask
        seg_candidates = [
            os.path.join(patient_dir, "segmentation.nii.gz"),
            os.path.join(patient_dir, "segmentation.nii"),
            os.path.join(patient_dir, "*seg*.nii.gz"),
        ]

        y = None
        for candidate in seg_candidates:
            if "*" in candidate:
                matches = glob.glob(candidate)
                if matches:
                    y = self.load_nifti(matches[0])
                    break
            elif os.path.exists(candidate):
                y = self.load_nifti(candidate)
                break

        if y is None:
            raise FileNotFoundError(f"Could not find segmentation for {patient_id}")

        # Resize segmentation to match modalities
        y = self.pad_volume(y, self.patch_size).astype(np.int64)

        # Fix missing class: map 4 → 3
        y[y == 4] = 3

        return {
            "image": torch.from_numpy(x),
            "mask": torch.from_numpy(y),
            "patient_id": patient_id,
        }


if __name__ == "__main__":
    # Test the dataset
    dataset = BrainTumorVolume3D("./data/upenn_gbm_nifti")
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample from {sample['patient_id']}:")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Mask classes: {torch.unique(sample['mask']).tolist()}")
