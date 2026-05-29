# Multi-GPU Training Strategy for Your Hardware

## Your Hardware Inventory

```
RTX 3050         → 6GB VRAM (primary GPU)
Intel UHD        → 11GB shared VRAM (integrated GPU)
System RAM       → 24GB
─────────────────────────────────────
Total GPU VRAM   → 17GB effective
Total Memory     → 41GB available
```

## Strategic Options

### ⚠️ Option A: Sequential Training (SAFE - Recommended)
Train modalities one at a time, reuse GPU memory.

```
Week 2: Chest X-ray (ResNet50)          → Save best model
        ↓ Free GPU memory
Week 3: CT Segmentation (3D UNet)       → Save best model
        ↓ Free GPU memory
Week 4: Other modalities...
```

**Pros:**
- ✓ No complexity, works with current code
- ✓ Each model gets full GPU attention (faster training)
- ✓ 17GB is plenty for ResNet50 or single 3D UNet
- ✓ Safe - no coordination issues

**Cons:**
- ✗ Serial (no parallelism)
- ✗ Takes 2-3x longer (8 weeks → 16 weeks)

**Verdict:** Best if you have time.

---

### 🔥 Option B: Model Parallel (Distributed Chunks)
Split ONE model across both GPUs during training.

```
RTX 3050 (6GB)           Intel UHD (11GB)
┌──────────────────┐  ┌──────────────────┐
│ Encoder Blocks   │←→│ Decoder Blocks    │
│ (Input → Middle) │  │ (Middle → Output) │
└──────────────────┘  └──────────────────┘
     ↓                       ↓
  Batch Split:        Batch Recombined
  (32 → 16 per GPU)   (Gradients synced)
```

**Ideal for:** 3D U-Net (large volumetric models)

**Implementation:**
```python
model = UNet3D(...)

# Split layers across GPUs
model.encoder = nn.DataParallel(model.encoder, device_ids=[0, 1])
model.decoder = nn.DataParallel(model.decoder, device_ids=[1, 0])
```

**Pros:**
- ✓ Run large models that don't fit on one GPU
- ✓ Both GPUs active simultaneously
- ✓ Better memory utilization

**Cons:**
- ✗ Inter-GPU bandwidth becomes bottleneck (PCIe 3.0)
- ✗ ~20-30% overhead from GPU-to-GPU communication
- ✗ Complex debugging

**Verdict:** Good for 3D models, overhead acceptable.

---

### ⚡ Option C: Data Parallel (Best for You)
Same model on both GPUs, different data chunks.

```
Batch: 64 images
  ↓
Split: 32 to RTX 3050, 32 to Intel UHD
  ↓
Forward pass in parallel
  ↓
Gradients averaged
  ↓
Weight update
```

**Implementation:**
```python
model = UNet3D(...)
model = nn.DataParallel(model, device_ids=[0, 1])

# Training loop stays the same
for batch in loader:
    outputs = model(batch)  # Automatically split across GPUs
    loss.backward()
    optimizer.step()
```

**Pros:**
- ✓ Easiest to implement
- ✓ Works with ResNet50 AND 3D UNet
- ✓ Near-linear speedup (2x GPUs ≈ 1.7-1.8x faster)
- ✓ Batch parallelism is natural

**Cons:**
- ✗ Requires same memory per GPU (can't use UHD's extra 5GB efficiently)
- ✗ RTX 3050 becomes bottleneck (slower peer)

**Verdict:** Best compromise.

---

### 🚀 Option D: Hybrid Async Training (ADVANCED)
Train multiple modalities in parallel using different GPUs.

```
GPU 0 (RTX 3050)               GPU 1 (Intel UHD)
Week 2-3: Chest X-ray          Week 2-3: CT Segmentation
ResNet50 (small, fast)         3D UNet (large, slow)
Finish in 1 week               Finish in 3 weeks
           ↓
Week 4: Swap - train other      
modality on freed GPU
```

**Pros:**
- ✓ Maximizes GPU utilization
- ✓ Train 2 models simultaneously
- ✓ Finish in ~5 weeks instead of 8

**Cons:**
- ✗ Requires separate training scripts
- ✗ Memory careful - RTX 3050 only 6GB
- ✗ Complex to manage

**Verdict:** Best if you want speed AND can manage complexity.

---

## Recommendation: Hybrid Approach

### Strategy: Option C (Data Parallel) + Option B (for 3D models)

**Week 2-3: Chest X-ray on Data Parallel**
```python
batch_size = 32  # Total batch
# RTX 3050: 16 images, Intel UHD: 16 images
# Expected: ~1.7x faster than single GPU
# Time: ~8 hours total (vs 14 hours single GPU)
```

**Week 3-4: 3D U-Net with Model Parallel**
```python
# Too big for one GPU? Split model
encoder → RTX 3050
decoder → Intel UHD
# ~20-30% overhead but works
# Time: ~40 hours (vs 60+ hours single GPU)
```

**Week 5+: Continue with proven approach**

---

## Implementation Guide

### Step 1: Detect Both GPUs

```python
import torch

print(f"CUDA devices: {torch.cuda.device_count()}")
print(f"CUDA device names:")
for i in range(torch.cuda.device_count()):
    print(f"  [{i}] {torch.cuda.get_device_name(i)}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Output expected:
# CUDA devices: 2
# CUDA device names:
#   [0] NVIDIA RTX 3050
#   [1] Intel Arc/UHD Graphics
# CUDA available: True
```

### Step 2: Enable Data Parallel Training

```python
# train_3d_segmentation.py modification

model = UNet3D(in_channels=2, out_channels=4, features=32)

# Auto-detect and use all GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)  # No device_ids = use all

model = model.to(device)  # device = "cuda"
```

### Step 3: Adjust Batch Size

With 2 GPUs, you can increase batch size:

```python
# Single GPU (6GB RTX 3050)
batch_size = 2

# Dual GPU (6GB + 11GB)
batch_size = 4  # Each GPU gets 2 images
# Or batch_size = 6 (RTX gets 3, UHD gets 3)
```

### Step 4: Monitor Both GPUs

```bash
# Terminal 1: Watch GPU usage
watch -n 1 nvidia-smi

# Expected output:
# GPU 0 (RTX 3050): 85% utilization, 5.2GB/6GB
# GPU 1 (UHD): 60% utilization, 6.8GB/11GB
```

---

## Memory Math

### Chest X-ray (ResNet50) - Data Parallel

```
Single GPU (RTX 3050):
- Model weights: 0.1GB
- Batch 32 images: 0.8GB
- Activations: 1.2GB
- Gradients: 0.1GB
─────────────────────
Total: 2.2GB → Fits!

With batch_size=64 (32 per GPU):
- Per GPU: 1.1GB
- RTX 3050: 1.2GB (fits with headroom)
- Intel UHD: 1.2GB (fits with 9.8GB free)
```

### 3D U-Net (64×128×128) - Model Parallel

```
Single GPU (RTX 3050):
- Model: 0.2GB
- Batch 2 volumes: 4.5GB
- Activations: 1.0GB
─────────────────────
Total: 5.7GB → TIGHT! (only 0.3GB free)

With Model Parallel:
- Encoder on RTX: 0.1GB model + 2.3GB batch + 0.5GB activations = 2.9GB ✓
- Decoder on UHD: 0.1GB model + 2.3GB batch + 0.5GB activations = 2.9GB ✓
- PCIe transfer: ~100ms per batch (acceptable overhead)
```

---

## Performance Estimates

### Option A: Sequential (Safe)
- Weeks 2-3: Chest X-ray (1 week)
- Weeks 4-6: CT (3 weeks)
- Total: 8+ weeks
- GPU utilization: 70% (one GPU per model)

### Option C: Data Parallel (Recommended)
- Weeks 2-3: Chest X-ray on 2 GPUs (0.6 weeks)
- Weeks 3-4: CT on 2 GPUs (1.8 weeks)
- Total: 5 weeks
- GPU utilization: 85% (both GPUs always busy)
- Speedup: ~1.7x per model, 3.2x total

### Option D: Async Hybrid (Fast)
- Weeks 2-3: Chest X-ray on GPU0 + CT on GPU1 (parallel!)
- Total: 4 weeks
- GPU utilization: 95% (everything in parallel)
- Speedup: 2x

---

## My Recommendation

**Use Option C (Data Parallel) for your RTX 3050 setup:**

### Why:
1. ✓ Your UHD has extra VRAM (11GB) that goes unused otherwise
2. ✓ Data parallelism is natural for medical imaging
3. ✓ Simple to implement (3 lines of code change)
4. ✓ Near-linear speedup (1.7-1.8x for 2 GPUs)
5. ✓ Only ~20-30% slower than dedicated dual-GPU setup

### Modified Training Config:

```python
CONFIG = {
    # ... existing ...
    "batch_size": 4,                    # Increased from 2
    "use_data_parallel": True,          # NEW
    "pin_memory": True,                 # Keep for CUDA
    "num_workers": 4,                   # Increase prefetch
}
```

### Expected Results:

**Week 2-3: Chest X-ray**
- Sequential time: 14 hours
- Data parallel time: ~8 hours (1.7x speedup)
- Time saved: 6 hours

**Week 3-4: 3D U-Net**
- Sequential time: 60 hours
- Data parallel time: ~36 hours (1.7x speedup)
- Time saved: 24 hours

**Total savings: 30 hours = 1.25 days → finish in 4 weeks instead of 5**

---

## Code Changes Required

Create new file: `train_multi_gpu.py` (wrapper around existing code)

```python
import torch
import torch.nn as nn
from train_3d_segmentation import train  # Reuse existing

# Auto-enable data parallel
def setup_multi_gpu():
    if torch.cuda.device_count() > 1:
        print(f"🚀 Detected {torch.cuda.device_count()} GPUs")
        print(f"   GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 1: {torch.cuda.get_device_name(1)}")
        return True
    return False

if __name__ == "__main__":
    has_multi_gpu = setup_multi_gpu()
    CONFIG["use_data_parallel"] = has_multi_gpu
    train()
```

---

## Summary

| Strategy | Speed | Complexity | Recommended? |
|----------|-------|-----------|--------------|
| **A: Sequential** | 1x | ⭐ | Good if no time pressure |
| **B: Model Parallel** | 1.4x | ⭐⭐⭐⭐ | Complex, not worth it |
| **C: Data Parallel** | 1.7x | ⭐⭐ | **YES ✅** |
| **D: Async Hybrid** | 2x | ⭐⭐⭐ | Only if managing multiple scripts |

**Final Answer: Use Option C (Data Parallel) - 1.7x speedup with minimal code changes** 🚀

