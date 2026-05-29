# 3D Volumetric Brain Tumor Segmentation Pipeline

This is a complete, production-ready pipeline for segmenting brain tumors using 3D U-Net.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    3D U-Net Architecture                    │
├──────────────────┬──────────────────┬──────────────────────┤
│                  │                  │                      │
│   Encoder        │   Bottleneck     │    Decoder           │
│  (Contract)      │                  │   (Expand)           │
│                  │                  │   + Skip Connections │
│ Conv → Pool      │   Conv Block     │   UpConv → Cat       │
│ Conv → Pool      │   (16x features) │   UpConv → Cat       │
│ Conv → Pool      │                  │   UpConv → Cat       │
│ Conv → Pool      │                  │   UpConv → Cat       │
│                  │                  │   Final Conv (4 cls) │
└──────────────────┴──────────────────┴──────────────────────┘

Input:  (batch, 2_channels, 64_depth, 128_height, 128_width)
Output: (batch, 4_classes, 64_depth, 128_height, 128_width)

Classes: 0=Background, 1=Necrotic Core, 2=Edema, 3=Enhancing Tumor
Modalities: T1ce (contrast-enhanced T1) + FLAIR (fluid-suppressed T2)
```

---

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Key additions:
- `pydicom` - Read DICOM files from IDC
- `nibabel` - Read/write NIfTI format
- `monai` - Optional but useful for medical imaging utilities

---

## Step 2: Convert DICOM to NIfTI (UPENN-GBM)

Your UPENN-GBM data is currently DICOM files from IDC. Convert them to organized NIfTI format:

```bash
python utils/dicom_to_nifti.py
```

This will:
- Read DICOM files from `data/upenn_gbm_data/upenn_gbm/`
- Convert to NIfTI (.nii.gz) format
- Organize into `data/upenn_gbm_nifti/` with patient folder structure

---

## Step 3: (Optional) Download BraTS2020

If you want to use the full BraTS2020 dataset (~50GB):

```python
# Create utils/download_brats.py
import os
from monai.apps import download_and_extract

download_and_extract(
    url="https://www.med.upenn.edu/cbica/brats2020/data.html",
    filepath="./data/brats2020",
    output_dir="./data/brats2020"
)
```

Or manually download from: https://www.synapse.org/#!Synapse:syn22015602/files/

---

## Step 4: Train the 3D U-Net

```bash
python train_3d_segmentation.py
```

Configuration options (edit at top of script):

```python
CONFIG = {
    "data_dir": "./data/upenn_gbm_nifti",  # or "./data/brats2020" 
    "epochs": 50,
    "batch_size": 2,          # Reduce if OOM errors
    "learning_rate": 1e-4,
    "patch_size": (64, 128, 128),  # Depth, Height, Width
}
```

**Expected training time:**
- 10 patients at batch_size=2: ~2-3 hours per epoch on M1 Pro
- With BraTS (250+ patients): ~30-50 hours for 50 epochs

**Output:**
- Best model checkpoint: `segmentation_models/unet3d_*.pth`
- Training history: `segmentation_models/history.json`
- Config: `segmentation_models/config.json`

---

## Key Differences from 2D Pipeline

| Aspect | 2D Slices | 3D Volumes |
|--------|-----------|-----------|
| Input | Single 2D image (128×128) | Full 3D volume (64×128×128) |
| Channels | 1 (single modality) | 2 (T1ce + FLAIR) |
| Memory | ~500MB for batch=16 | ~4GB for batch=2 |
| Context | No depth awareness | Full 3D spatial context |
| Accuracy | Dice ~0.46-0.64 | Dice ~0.65-0.75 (expected) |
| Training | Fast but limited | Slower but much better |

---

## Data Organization

Expected structure after conversion:

```
data/upenn_gbm_nifti/
├── UPENN-GBM-00005/
│   ├── T1ce.nii.gz
│   ├── FLAIR.nii.gz
│   ├── segmentation.nii.gz
│   └── (other modalities)
├── UPENN-GBM-00007/
│   ├── T1ce.nii.gz
│   ├── FLAIR.nii.gz
│   ├── segmentation.nii.gz
├── ...
```

Dataset loader will:
1. Find all patient folders
2. Load T1ce + FLAIR
3. Normalize to [0, 1]
4. Resize/crop to 64×128×128
5. Load segmentation mask
6. Stack as (batch, 2, 64, 128, 128) for model input

---

## Model Architecture Details

**3D U-Net Specifics:**
- Encoder: 4 downsampling blocks (each reduces size by 2×)
- Bottleneck: 16× feature channels (512 in base config)
- Decoder: 4 upsampling blocks with skip connections
- Parameters: ~21.7M trainable parameters
- Total input: 2 channels (modalities) × 64×128×128 = ~2.1M values per volume

**Loss Function:**
- Combination of:
  - **Cross-Entropy Loss**: Standard multi-class loss
  - **Dice Loss**: Emphasizes overlap between predictions and ground truth
  - Total: `loss = CE_loss + Dice_loss`

**Metrics:**
- **Dice Coefficient**: 0.5-1.0 where 1.0 is perfect overlap
- **Per-class Dice**: Separate scores for each tumor component

---

## Next Steps

1. **Run DICOM conversion:**
   ```bash
   python utils/dicom_to_nifti.py
   ```

2. **Test dataset loading:**
   ```bash
   python utils/dataset_3d.py
   ```

3. **Start training:**
   ```bash
   python train_3d_segmentation.py
   ```

4. **Monitor progress:**
   - Check `segmentation_models/history.json` for loss/dice curves
   - Watch console output for validation metrics

---

## Troubleshooting

**OOM (Out of Memory) errors:**
- Reduce `batch_size` (try 1 instead of 2)
- Reduce `patch_size` (try 48×96×96 instead of 64×128×128)

**DICOM conversion issues:**
- Ensure all DICOM files are readable (check metadata with pydicom)
- Some series might be scout images — skip them automatically

**Missing modalities:**
- Dataset loader is flexible with naming
- Edit `dataset_3d.py` to match your naming convention

---

## References

- U-Net 3D: https://arxiv.org/abs/1606.06650
- BraTS Challenge: https://www.med.upenn.edu/cbica/brats2020/
- MONAI: https://monai.io/
