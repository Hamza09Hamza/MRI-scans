#!/usr/bin/env python3
"""
Convert DICOM files from IDC to NIfTI format.
Handles hierarchical directory structure from IDC downloads.
"""

import os
import glob
from pathlib import Path
import pydicom
import nibabel as nib
import numpy as np
from tqdm import tqdm


def find_dicom_series(patient_dir):
    """
    Find all DICOM series in a patient directory.
    Returns dict: {series_name: [list of DICOM files]}
    """
    series_dict = {}

    # Traverse the complex IDC directory structure
    for root, dirs, files in os.walk(patient_dir):
        dicom_files = [f for f in files if f.endswith('.dcm')]

        if dicom_files:
            # Infer series name from path or DICOM metadata
            try:
                dcm = pydicom.dcmread(os.path.join(root, dicom_files[0]))
                series_name = getattr(dcm, 'SeriesDescription', 'Unknown').replace(' ', '_')
                series_dict[series_name] = [os.path.join(root, f) for f in dicom_files]
            except Exception as e:
                print(f"Error reading DICOM metadata: {e}")
                series_dict[f"Series_{len(series_dict)}"] = [os.path.join(root, f) for f in dicom_files]

    return series_dict


def dcm_series_to_nifti(dicom_files, output_path):
    """
    Convert a series of DICOM files to a single NIfTI volume.
    """
    try:
        # Read first DICOM to get metadata
        ds_first = pydicom.dcmread(dicom_files[0])

        # Read all DICOM files
        datasets = []
        for dcm_file in sorted(dicom_files):
            ds = pydicom.dcmread(dcm_file)
            datasets.append(ds)

        # Stack pixel arrays (z, y, x)
        pixel_array = np.stack([ds.pixel_array.astype(np.float32) for ds in datasets], axis=0)

        # Create affine (identity for now, proper DICOM → world conversion would be more complex)
        affine = np.eye(4)

        # Create NIfTI image
        nifti_img = nib.Nifti1Image(pixel_array, affine)
        nib.save(nifti_img, output_path)

        return True
    except Exception as e:
        print(f"Error converting DICOM series: {e}")
        return False


def convert_upenn_gbm_to_nifti(upenn_gbm_dir, output_dir):
    """
    Convert UPENN-GBM DICOM files from IDC to organized NIfTI structure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all patient directories
    patient_dirs = glob.glob(os.path.join(upenn_gbm_dir, "UPENN-GBM-*"))

    for patient_dir in tqdm(patient_dirs, desc="Converting UPENN-GBM"):
        patient_id = os.path.basename(patient_dir)
        patient_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)

        # Find all DICOM series
        series_dict = find_dicom_series(patient_dir)

        for series_name, dicom_files in series_dict.items():
            if not dicom_files:
                continue

            output_path = os.path.join(patient_output_dir, f"{series_name}.nii.gz")

            if not os.path.exists(output_path):
                success = dcm_series_to_nifti(dicom_files, output_path)
                if success:
                    print(f"✓ {patient_id}/{series_name}")
                else:
                    print(f"✗ {patient_id}/{series_name} - conversion failed")


if __name__ == "__main__":
    upenn_gbm_dir = "./data/upenn_gbm_data/upenn_gbm"
    output_dir = "./data/upenn_gbm_nifti"

    print("Converting UPENN-GBM DICOM to NIfTI...")
    convert_upenn_gbm_to_nifti(upenn_gbm_dir, output_dir)
    print(f"\nConversion complete! Output: {output_dir}")
