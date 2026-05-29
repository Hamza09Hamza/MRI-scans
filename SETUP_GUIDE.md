# 3D Brain Tumor Segmentation - Setup & Training Guide

## Quick Start

This is a complete production-ready 3D U-Net pipeline for multi-modal brain tumor segmentation using BraTS2020 and UPENN-GBM datasets.

### System Requirements

**M1 Pro (16GB Unified Memory):**
- Batch size: 2
- Training time: ~30-50 hours for 50 epochs
- VRAM usage: ~4GB per 3D volume batch

**RTX 3050 (6GB VRAM, 24GB RAM):**
- Batch size: 1 (tight fit) or 2 (with gradient checkpointing)
- Training time: ~50-70 hours for 50 epochs
- Recommended: Use batch_size=1 for stability

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/Hamza09Hamza/MRI-scans.git
cd MRI-scans
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- `torch==2.11.0` - Deep learning framework
- `monai==1.3.0` - Medical imaging utilities
- `nibabel==5.4.2+` - NIfTI file handling
- `pydicom==2.4.4` - DICOM conversion
- `scikit-learn==1.7.2` - Metrics
- `tqdm>=4.65.0` - Progress bars

---

## Data Setup

### Option A: BraTS2020 Only (Recommended for Testing)

1. **Download BraTS2020:**
   - Visit: https://www.synapse.org/#!Synapse:syn22015602/files/
   - Extract to: `data/BraTS2020/`
   - Structure:
     ```
     data/BraTS2020/
     ├── BraTS2020_TrainingData/
     │   └── MICCAI_BraTS2020_TrainingData/
     │       ├── BraTS20_Training_001/
     │       └── ...
     └── BraTS2020_ValidationData/
         └── MICCAI_BraTS2020_ValidationData/
             ├── BraTS20_Validation_001/
             └── ...
     ```

2. **Verify dataset loads:**
   ```bash
   python3 test_combined_dataset.py
   ```

### Option B: BraTS2020 + UPENN-GBM (Full Pipeline)

1. **Download BraTS2020** (see Option A)

2. **Download UPENN-GBM from IDC:**
   - Use: https://github.com/ImagingDataCommons/idc-index
   - Extract DICOM files to: `data/upenn_gbm_data/upenn_gbm/`

3. **Convert DICOM to NIfTI:**
   ```bash
   python3 utils/dicom_to_nifti.py
   ```
   Output: `data/upenn_gbm_nifti/`

4. **Organize UPENN-GBM files:**
   ```bash
   python3 utils/organize_upenn_nifti.py
   ```
   Creates standard naming: `T1ce.nii.gz`, `FLAIR.nii.gz`, `segmentation.nii.gz`

5. **Verify combined dataset:**
   ```bash
   python3 test_combined_dataset.py
   ```
   Expected output:
   - BraTS2020 training: 338 patients (with segmentation)
   - BraTS2020 validation: 125 patients
   - UPENN-GBM: 5 patients
   - **Total: 374 volumes**

---

## Training

### M1 Pro Configuration (Default)

```bash
# Uses automatic MPS device detection
python3 train_3d_segmentation.py
```

**Default Config:**
- Device: MPS (automatic)
- Batch size: 2
- Learning rate: 1e-4
- Epochs: 50
- Modalities: t1ce, flair
- Patch size: 64×128×128

### RTX 3050 Configuration

**Option 1: Single GPU (Recommended)**

Edit `train_3d_segmentation.py`, line 29-43:
```python
CONFIG = {
    "brats_dir": "./data/BraTS2020",
    "upenn_dir": "./data/upenn_gbm_nifti",
    "output_dir": "./segmentation_models",
    "device": "cuda",  # Force CUDA
    "epochs": 50,
    "batch_size": 1,   # Reduce for 6GB VRAM
    "learning_rate": 1e-4,
    "num_workers": 2,
    "pin_memory": True,
    # ... rest of config
}
```

Then run:
```bash
python3 train_3d_segmentation.py
```

**Option 2: With Gradient Checkpointing (if OOM)**

Add to `train_3d_segmentation.py` after model creation:
```python
from torch.utils.checkpoint import checkpoint
model.gradient_checkpointing_enable()
```

---

## Architecture Overview

**3D U-Net:**
- Input: (batch, 2_channels, 64, 128, 128)
- Output: (batch, 4_classes, 64, 128, 128)
- Parameters: 22.6M
- Classes: Background (0), Necrotic Core (1), Edema (2), Enhancing Tumor (3)

**Loss Function:**
- Cross-Entropy Loss + Dice Loss

**Metrics:**
- Dice Score per class (target: 0.65-0.75+)

---

## Output Files

After training completes, find results in `segmentation_models/`:

```
segmentation_models/
├── config.json                               # Training configuration
├── history.json                              # Loss/Dice curves
├── unet3d_multimodal_epoch01_dice0.5234.pth # Checkpoint (best)
├── unet3d_multimodal_epoch02_dice0.5456.pth
└── ...
```

**Analyze results:**
```python
import json
with open('segmentation_models/history.json') as f:
    history = json.load(f)
    
print(f"Best Dice: {max(history['val_dice']):.4f}")
print(f"Final Loss: {history['val_loss'][-1]:.4f}")
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA out of memory | Batch size too large | Reduce `batch_size` to 1 |
| MPS crashes on M1 | Device compatibility | Set `device: "cpu"` temporarily |
| Missing segmentation files | BraTS dataset incomplete | Re-download or check extraction |
| Dimension mismatch in padding | NIfTI with extra dims | Already handled - update nibabel |
| Data loading too slow | num_workers too high | Reduce `num_workers` to 0-2 |

---

## Expected Performance

**On M1 Pro (batch_size=2):**
- ~2-3 min/epoch
- 50 epochs: 1.5-2.5 hours total
- Dice score convergence: epoch 15-20
- Expected final Dice: 0.65-0.75

**On RTX 3050 (batch_size=1):**
- ~3-5 min/epoch (slower due to batch size)
- 50 epochs: 2.5-4 hours total
- Dice score convergence: epoch 15-25
- Expected final Dice: 0.63-0.72

---

## Multi-Dataset Support

The unified loader (`utils/multi_dataset_3d.py`) automatically:
- Handles BraTS2020 naming convention (lowercase modalities: t1ce, flair)
- Handles UPENN-GBM naming convention (capitalized: T1ce, FLAIR)
- Filters incomplete patients (missing segmentation)
- Performs robust padding/cropping for mixed dimensions
- Removes extra NIfTI dimensions (time, singleton dims)

---

## Next Steps

1. **Download data** (BraTS2020 minimum recommended)
2. **Run test:** `python3 test_combined_dataset.py`
3. **Adjust batch_size** for your GPU
4. **Start training:** `python3 train_3d_segmentation.py`
5. **Monitor:** Check `segmentation_models/history.json` during training

---

## References

- **U-Net 3D:** https://arxiv.org/abs/1606.06650
- **BraTS Challenge:** https://www.med.upenn.edu/cbica/brats/
- **MONAI Library:** https://monai.io/
- **PyTorch Lightning:** https://www.pytorchlightning.ai/

