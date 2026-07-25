# Brain MRI Tumour Classification

This directory documents the brain MRI classification work in the `MRI-scans` repository.

The project trains a ResNet50-based image classifier to distinguish four classes:

- **No tumour**
- **Pituitary tumour**
- **Meningioma**
- **Glioma**

> **Research prototype only.** The model is not clinically validated and must not be used for diagnosis, triage, treatment, or any other patient-care decision.

## Current confirmed evaluation statement

The currently confirmed aggregate result for the project is:

| Item | Confirmed value |
|---|---|
| Evaluation size | **3,482 images** |
| Evaluation composition | **Combined images from two datasets used by the project** |
| Overall accuracy | **Approximately 97%** |

This should be described as a **combined two-dataset evaluation**, not as a purely external hospital validation.

### Why the previous documentation was changed

The previous README mixed information from different evaluation artifacts:

- `hospital_eval/metrics.json` records a legacy hospital-only run containing **1,857 images** and an accuracy near 97%.
- The confusion matrix printed in the old README contained **3,482 images**.
- That confusion matrix and the surrounding accuracy/per-class table did not describe one internally consistent experiment.

To avoid publishing false precision, the contradictory confusion matrix and per-class table have been removed. The repository now reports only the confirmed aggregate result above until the current 3,482-image combined evaluation is rerun and exports one matching set of metrics.

## Model architecture

The current model implementation is defined in [`../utils/model.py`](../utils/model.py).

```text
Input image: 128 x 128 RGB
        |
        v
ResNet50 backbone pretrained on ImageNet
        |
        v
2048-dimensional feature vector
        |
        v
Linear 2048 -> 1024
BatchNorm + ReLU + Dropout
        |
        v
Linear 1024 -> 512
BatchNorm + ReLU + Dropout
        |
        v
Linear 512 -> 256
BatchNorm + ReLU + Dropout
        |
        v
Linear 256 -> 4 class logits
```

The configuration in [`../config.json`](../config.json) currently specifies:

- 128 x 128 input images
- four output classes
- batch size 32
- learning rate `0.0001`
- dropout rate `0.5`
- up to 80 epochs
- Adam optimization
- `ReduceLROnPlateau` scheduling

These are implementation settings, not proof that the model is clinically robust.

## Data sources

The project has used two image sources during its development:

- a public Kaggle brain-tumour classification dataset
- the Epic & CSCR Hospital dataset used by the repository's hospital-evaluation workflow

The current 3,482-image headline combines images from two sources. Before the result is used in a paper, CV, portfolio, or comparison table, the repository should additionally record:

- exact dataset versions and download locations
- dataset licenses and permitted uses
- exact image counts contributed by each source
- patient counts, where available
- whether multiple slices belong to the same patient
- duplicate-image and near-duplicate checks
- any overlap between the two datasets
- the exact train, validation, and evaluation split logic

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Review [`../SETUP_GUIDE.md`](../SETUP_GUIDE.md) for the broader environment setup.

## Training

The primary training entry point is:

```bash
python train.py
```

The training configuration is stored in `config.json`. Checkpoints are written under the configured output directory.

To resume training where supported by the script:

```bash
python train.py --resume latest
```

Before training, confirm that the dataset directories match the paths in `config.json` and that no patient or duplicate image appears across training and evaluation partitions.

## Evaluation

The repository contains multiple evaluation paths:

- [`../evaluation/evaluate.py`](../evaluation/evaluate.py) for general evaluation work
- [`../evaluation/evaluate_hospital.py`](../evaluation/evaluate_hospital.py) for the legacy hospital-only workflow

The hospital-only script writes a metrics JSON file and confusion matrix for the dataset it is given. Its existing `hospital_eval/metrics.json` belongs to the older 1,857-image run and should not be presented as the export for the current combined 3,482-image evaluation.

A corrected combined evaluation should save all of the following from one run:

```text
evaluation source and version
model checkpoint hash
configuration
image count
patient count, when available
class distribution
overall accuracy
balanced accuracy
per-class precision/recall/F1
sensitivity and specificity
confusion matrix
calibration results
misclassified examples
```

The total in the confusion matrix must equal the documented evaluation count, and all summary metrics must be calculated from that same prediction set.

## Inference example

The model expects RGB input normalized with ImageNet statistics. A simplified example is:

```python
from PIL import Image
import torch
from torchvision import transforms

from MRI.load_model import load_brain_classifier

model, device = load_brain_classifier()
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

image = Image.open("brain_mri.jpg").convert("RGB")
batch = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(batch)
    probabilities = torch.softmax(logits, dim=1)

classes = ["glioma", "meningioma", "notumor", "pituitary"]
predicted_index = probabilities.argmax(dim=1).item()
print(classes[predicted_index], probabilities[0, predicted_index].item())
```

Confirm the class order against the dataset loader and checkpoint used for the run. A class-order mismatch can produce apparently valid but incorrect predictions.

## What the reported accuracy does not prove

Approximately 97% accuracy on a combined image set does not, by itself, prove:

- clinical safety
- patient-level generalization
- performance on another hospital or scanner
- absence of train/test leakage
- robustness to different MRI sequences
- reliable probability calibration
- acceptable false-negative rates
- readiness for deployment

Image-level accuracy can be inflated when several related slices from one patient are split across training and testing. Medical evaluation should therefore use patient-level separation whenever patient identifiers are available.

## Recommended next evaluation

The next trustworthy evaluation should:

1. Freeze one model checkpoint and configuration.
2. Build a manifest containing every image, source dataset, class, and patient identifier where available.
3. Detect exact and perceptual duplicates across all partitions.
4. Split by patient rather than by individual image.
5. Keep one dataset or institution completely external when possible.
6. Export one machine-readable metrics file and one matching confusion matrix.
7. Report sensitivity, specificity, precision, recall, F1, calibration, and confidence intervals.
8. Review representative false positives and false negatives.

## Segmentation work

This repository also contains separate segmentation experiments. See:

- [`../SEGMENTATION_SETUP.md`](../SEGMENTATION_SETUP.md)
- [`../SEGMENTATION/README.md`](../SEGMENTATION/README.md)
- [`../train_3d_segmentation.py`](../train_3d_segmentation.py)

Classification and segmentation results must be evaluated separately. A good classification score does not establish accurate tumour localization or volume measurement.

## Responsible use

This project is intended for education and research. Medical images and metadata must be handled according to the applicable authorization, privacy, security, and data-governance requirements.

Do not describe this model as production-ready, clinical-grade, or externally validated unless those claims are supported by a documented study and review process.
