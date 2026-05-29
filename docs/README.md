# Brain Tumor Classification CNN

A production-ready CNN for classifying brain tumors in MRI scans with GPU/MPS acceleration, hyperparameter grid search, and full evaluation pipeline.

## Project Structure

```
├── config.json              # Configuration file (hyperparameters)
├── train.py                 # Main training script
├── grid_search.py           # Hyperparameter grid search
├── evaluate.py              # Model evaluation and metrics
├── utils/
│   ├── model.py            # CNN architecture
│   ├── dataset.py          # Data loading and augmentation
│   └── training_utils.py   # Training & evaluation functions
├── models/                 # Saved weights & results directory
├── Training/               # Training dataset
└── Testing/                # Test dataset
```

## Quick Start

### 1. Single Training Run
```bash
python train.py --config config.json --output models
```

**What it does:**
- Loads hyperparameters from `config.json`
- Trains CNN for N epochs with GPU/MPS optimization
- Saves best model to `models/best_model.pth`
- Saves training history and config to `models/`

### 2. Hyperparameter Grid Search
```bash
python grid_search.py
```

**What it does:**
- Tests all combinations of:
  - Learning rates: [1e-4, 1e-3, 1e-2]
  - Batch sizes: [32, 64, 128]
  - Dropout rates: [0.2, 0.4, 0.6]
- Creates `grid_search_results_YYYYMMDD_HHMMSS/` directory
- Each trial gets its own subdirectory with results
- Saves summary ranking all trials by test accuracy

### 3. Evaluate Model
```bash
python evaluate.py --weights models/best_model.pth --config config.json
```

**What it does:**
- Loads saved model weights
- Evaluates on test set
- Prints per-class metrics
- Saves confusion matrix plot
- Saves evaluation metrics JSON

## Configuration (config.json)

All hyperparameters are in `config.json`:

| Section | Key | Purpose |
|---------|-----|---------|
| `data` | `img_size` | Image resolution (128×128 recommended) |
| `training` | `batch_size` | Larger = better GPU utilization (32-256) |
| `training` | `learning_rate` | Optimizer step size (1e-4 to 1e-2) |
| `training` | `dropout_rate` | Regularization (0.2-0.6) |
| `training` | `epochs` | Training iterations |
| `augmentation` | `*` | Data augmentation settings |
| `device` | | Auto-detect GPU/MPS or force CPU |

Edit `config.json` before training to change defaults.

## GPU/MPS Optimizations

The code automatically detects and optimizes for:

- **CUDA** (NVIDIA GPUs):
  - Mixed Precision (float16) - 1.5-2x speedup
  - cuDNN auto-tuner - finds optimal algorithms
  - Gradient scaling for stability

- **MPS** (Apple Silicon Macs):
  - Parallel data loading
  - Optimized operators
  - Memory-efficient batch processing

- **CPU** (fallback):
  - Full float32 precision
  - Slower but works everywhere

### Batch Size Tips
- **GPU with 4GB+ VRAM**: Use 128-256
- **GPU with 2GB+ VRAM**: Use 64-128
- **MPS (M1/M2)**: Use 64-128
- **CPU**: Use 32 (slower but fits in RAM)

Larger batches = better GPU utilization and faster convergence.

## Output Files

### After Training (`train.py`)
```
models/
├── best_model.pth        # Best model weights
├── config.json           # Config used for training
└── history.json          # Training curves (loss & acc per epoch)
```

### After Grid Search (`grid_search.py`)
```
grid_search_results_YYYYMMDD_HHMMSS/
├── summary.json          # Ranking of all trials
├── trial_01_lr1e-04_bs32_do0.2/
│   ├── best_model.pth
│   ├── result.json       # Trial metrics
│   └── config.json
├── trial_02_lr1e-04_bs32_do0.4/
│   └── ...
└── ... (more trials)
```

### After Evaluation (`evaluate.py`)
```
models/
├── eval_metrics.json     # Per-class accuracy, confusion matrix
└── confusion_matrix.png  # Visualization
```

## Example Workflow

```bash
# 1. Run a quick training with defaults
python train.py

# 2. Launch grid search to find best hyperparameters
python grid_search.py
# (Check grid_search_results_* for best trial)

# 3. Update config.json with best hyperparameters
# 4. Train again with optimized config
python train.py --config config.json

# 5. Evaluate final model
python evaluate.py --weights models/best_model.pth
```

## Why This Architecture?

### Convolutional Blocks
- **Conv2d**: Learns spatial patterns (edges, textures, shapes)
- **BatchNorm**: Stabilizes training, enables higher learning rates
- **ReLU**: Non-linearity, doesn't saturate (better gradients)
- **MaxPool2d**: Reduces spatial dims, makes features shift-invariant

### Global Average Pooling
- Replaces flatten() → fewer parameters → less overfitting
- More robust to slight spatial shifts in input

### Dropout
- Regularization: prevents co-adaptation of neurons
- Higher rate (0.4-0.6) for smaller datasets

## Hyperparameter Tuning Guide

| Parameter | Effect | Typical Range | Notes |
|-----------|--------|---------------|-------|
| **Learning Rate** | Step size in gradient descent | 1e-4 to 1e-2 | Too high: unstable. Too low: slow. |
| **Batch Size** | Examples per gradient update | 32 to 256 | Larger: faster, needs more VRAM |
| **Dropout** | Regularization strength | 0.2 to 0.6 | Higher: more regularization |
| **Epochs** | Training iterations | 10 to 50 | Stop when val_loss plateaus |

Grid search tries many combinations to find the best.

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| CUDA out of memory | Batch size too large | Reduce `batch_size` in config.json |
| Training very slow | CPU training | Use GPU/MPS (auto-detected) |
| Overfitting (train >> val acc) | Model too large or data too small | Increase `dropout_rate` or add augmentation |
| Validation acc not improving | Learning rate too low | Increase `learning_rate` or use grid search |

## Citation

If you use this code, cite:
```
@software{mri_classifier_2024,
  title={Brain Tumor Classification CNN},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/mri-scans}
}
```

## License

MIT License - feel free to use and modify.
