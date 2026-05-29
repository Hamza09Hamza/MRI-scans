# Model Weights - Brain Tumor Classifier

This directory contains pre-trained model weights for the brain tumor classification model.

## File

```
brain_tumor_classifier_best.pth   (~100MB)
```

## Download Options

### Option 1: GitHub Releases (Recommended)

```bash
cd MRI/weights
wget https://github.com/Hamza09Hamza/MRI-scans/releases/download/v1.0/brain_tumor_classifier_best.pth
```

Or with curl:
```bash
curl -L https://github.com/Hamza09Hamza/MRI-scans/releases/download/v1.0/brain_tumor_classifier_best.pth \
  -o brain_tumor_classifier_best.pth
```

### Option 2: Train Your Own

```bash
# From repository root
python3 train.py

# Copy trained weights
cp models/brain_tumor_classifier_best.pth MRI/weights/brain_tumor_classifier_best.pth
```

## Model Specifications

- **Architecture:** ResNet50 with custom classifier head
- **Parameters:** 26M (1M trainable)
- **Input:** 128×128 RGB images
- **Output:** 4 classes (Normal, Pituitary, Meningioma, Glioma)
- **Accuracy:** 97.36% on hospital validation data
- **File Size:** ~100MB
- **Framework:** PyTorch

## Using Weights

```python
from MRI.load_model import load_brain_classifier

model, device = load_brain_classifier()
# Automatically loads from: MRI/weights/brain_tumor_classifier_best.pth
```

## Training Data

- **Primary:** Kaggle Brain Tumor Dataset (~3,000 images)
- **Validation:** Epic & CSCR Hospital Dataset (1,857 images)
- **Accuracy:** 97.36%

## For Developers

### After Retraining

1. Train the model
```bash
python3 train.py
```

2. Copy weights
```bash
cp models/brain_tumor_classifier_best.pth MRI/weights/brain_tumor_classifier_best.pth
```

3. Create GitHub release
```bash
git tag -a v1.0 -m "Brain tumor classifier v1.0"
git push origin v1.0
```

4. Upload to GitHub releases (via web UI)
   - https://github.com/Hamza09Hamza/MRI-scans/releases/new
   - Upload the .pth file

5. Update download URL in this file and QUICKSTART.md

## Validation Results

| Metric | Value |
|--------|-------|
| Accuracy | 97.36% |
| Precision | 97.8% |
| Recall | 97.9% |
| F1-Score | 97.8% |

Per-class:
- Normal: 99.7%
- Pituitary: 100%
- Meningioma: 99.1%
- Glioma: 92.7%

