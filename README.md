# MRI-scans

MRI-scans is a medical-imaging research repository focused primarily on brain MRI tumour classification, with additional experiments around segmentation, dataset preparation, and external evaluation.

The main classifier uses an ImageNet-pretrained ResNet50 backbone and predicts four classes:

- no tumour
- pituitary tumour
- meningioma
- glioma

> **Research status:** this project is not clinically validated and is not approved for diagnosis or patient care.

## Confirmed aggregate result

The currently confirmed project-level evaluation is:

- **3,482 images**
- a **combined evaluation set assembled from two datasets used by the project**
- approximately **97% overall classification accuracy**

This result must not be described as a purely external hospital validation because the 3,482-image evaluation mixes two data sources.

A previous README combined values from different evaluation runs: a legacy 1,857-image hospital-only metrics file and a separate 3,482-image confusion matrix. Those values did not describe one consistent experiment. The detailed classifier documentation has been corrected to report only the confirmed aggregate result until the current combined evaluation exports are regenerated.

## Documentation

- [`MRI/README.md`](MRI/README.md) — classifier architecture, evaluation statement, usage notes, and limitations
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — broader repository design notes
- [`SETUP_GUIDE.md`](SETUP_GUIDE.md) — environment setup
- [`SEGMENTATION_SETUP.md`](SEGMENTATION_SETUP.md) — segmentation experiments
- [`SEGMENTATION/README.md`](SEGMENTATION/README.md) — segmentation-specific documentation

## Main project areas

```text
MRI-scans/
├── train.py                         # classifier training entry point
├── grid_search.py                   # hyperparameter experiments
├── config.json                      # classifier configuration
├── utils/
│   ├── model.py                     # ResNet50-based classifier
│   ├── dataset.py                   # dataset utilities
│   ├── training_utils.py            # training helpers
│   ├── mask_model.py                # segmentation model helpers
│   └── dicom_to_nifti.py            # imaging conversion utility
├── evaluation/
│   ├── evaluate.py                  # evaluation utilities
│   └── evaluate_hospital.py         # legacy hospital-only evaluation script
├── MRI/                             # classifier documentation and loading helpers
└── SEGMENTATION/                    # segmentation research
```

## Evaluation integrity

Any future published result should record:

- exact dataset versions and licenses
- number of patients as well as number of images
- patient-level train/validation/test separation
- duplicate and cross-dataset overlap checks
- class distribution
- preprocessing and model checkpoint
- per-class sensitivity, specificity, precision, recall, and F1
- the confusion matrix generated from the same run
- confidence intervals and calibration

Until those artifacts are exported from the current 3,482-image combined run, the repository should use only the aggregate statement above.

## Responsible use

Do not use this code or its predictions for medical decisions. Medical-imaging models can fail because of scanner differences, acquisition protocols, image quality, dataset leakage, demographic shift, and label quality. All clinical use would require qualified medical review, privacy controls, external validation, and the applicable regulatory process.
