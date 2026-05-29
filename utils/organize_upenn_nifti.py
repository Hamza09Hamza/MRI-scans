#!/usr/bin/env python3
"""
Organize converted UPENN-GBM NIfTI files into standard format.
Renames series to standard modality names: T1ce, FLAIR, segmentation.
"""

import os
import glob
from pathlib import Path
from tqdm import tqdm

def organize_upenn_nifti(nifti_dir):
    """
    Organize UPENN-GBM NIfTI files to standard naming.
    """
    patient_dirs = sorted(glob.glob(os.path.join(nifti_dir, "UPENN-GBM-*")))

    for patient_dir in tqdm(patient_dirs, desc="Organizing UPENN-GBM NIfTI"):
        patient_id = os.path.basename(patient_dir)

        # Find all .nii.gz files in this patient directory
        nifti_files = glob.glob(os.path.join(patient_dir, "*.nii.gz"))

        # Map original names to standard names
        mappings = {}

        for filepath in nifti_files:
            filename = os.path.basename(filepath)
            lower_name = filename.lower()

            # Determine modality based on filename
            if "seg" in lower_name or "segmentation" in lower_name:
                target_name = "segmentation.nii.gz"
            elif "flair" in lower_name or "t2" in lower_name:
                target_name = "FLAIR.nii.gz"
            elif "t1" in lower_name and ("post" in lower_name or "contrast" in lower_name or "stealth" in lower_name):
                target_name = "T1ce.nii.gz"
            elif "t1" in lower_name:
                # Prefer post-contrast T1 if available, otherwise use this one
                continue  # Skip for now, prefer post-contrast version
            else:
                continue  # Skip unknown modalities

            # Only keep if not already mapped
            if target_name not in mappings:
                mappings[target_name] = filepath

        # Now handle the T1 case more carefully
        t1_files = [f for f in nifti_files if "t1" in os.path.basename(f).lower() and "seg" not in os.path.basename(f).lower()]

        if t1_files and "T1ce.nii.gz" not in mappings:
            # Prefer post-contrast T1
            post_contrast = [f for f in t1_files if "post" in os.path.basename(f).lower() or "stealth" in os.path.basename(f).lower()]
            if post_contrast:
                mappings["T1ce.nii.gz"] = post_contrast[0]
            elif t1_files:
                mappings["T1ce.nii.gz"] = t1_files[0]

        # Create symlinks or rename files
        for target_name, source_path in mappings.items():
            target_path = os.path.join(patient_dir, target_name)

            # Skip if target already exists
            if os.path.exists(target_path):
                continue

            # Create symlink
            try:
                os.symlink(os.path.basename(source_path), target_path)
            except Exception as e:
                print(f"Error creating symlink for {patient_id}/{target_name}: {e}")


if __name__ == "__main__":
    nifti_dir = "./data/upenn_gbm_nifti"

    print("Organizing UPENN-GBM NIfTI files...")
    organize_upenn_nifti(nifti_dir)
    print(f"\nOrganization complete!")

    # Verify
    print("\nVerifying patient directories:")
    patient_dirs = sorted(glob.glob(os.path.join(nifti_dir, "UPENN-GBM-*")))
    for pdir in patient_dirs:
        files = os.listdir(pdir)
        print(f"  {os.path.basename(pdir)}: {sorted(files)}")
