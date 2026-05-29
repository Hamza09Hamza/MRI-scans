# 🧠 Brain Tumor Classifier - Quick Start

**97.36% accuracy on hospital data**

## 1️⃣ Install (2 min)

```bash
git clone https://github.com/Hamza09Hamza/MRI-scans.git
cd MRI-scans
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 2️⃣ Load Model (30 sec)

```python
from MRI.load_model import load_brain_classifier

model, device = load_brain_classifier()
# ✓ Ready to classify!
```

## 3️⃣ Classify Image (1 line)

```python
from MRI.load_model import classify_image

result = classify_image("brain_mri.jpg", model, device)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Output Example

```
Prediction: Glioma
Confidence: 98.45%

All scores:
  Normal: 0.50%
  Pituitary: 0.20%
  Meningioma: 0.85%
  Glioma: 98.45%
```

---

## Model Specs

| Feature | Value |
|---------|-------|
| **Architecture** | ResNet50 |
| **Input** | 128×128 RGB images |
| **Classes** | Normal, Pituitary, Meningioma, Glioma |
| **Accuracy** | 97.36% (hospital validation) |
| **Speed** | 10ms/image (GPU) |
| **Parameters** | 26M total |

---

## Common Tasks

### Batch Process Multiple Images

```python
from pathlib import Path
from PIL import Image
import torch
from MRI.load_model import load_brain_classifier, preprocess_image

model, device = load_brain_classifier()

for img_path in Path("./scans").glob("**/*.jpg"):
    x = preprocess_image(str(img_path), device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    
    classes = ["Normal", "Pituitary", "Meningioma", "Glioma"]
    pred = classes[probs.argmax().item()]
    conf = probs.max().item()
    
    print(f"{img_path.name}: {pred} ({conf:.2%})")
```

### Save Predictions to CSV

```python
import csv
from pathlib import Path
from MRI.load_model import load_brain_classifier, classify_image

model, device = load_brain_classifier()

results = []
for img_path in Path("./scans").glob("**/*.jpg"):
    result = classify_image(str(img_path), model, device)
    results.append({
        "image": img_path.name,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "normal": result["all_scores"]["Normal"],
        "pituitary": result["all_scores"]["Pituitary"],
        "meningioma": result["all_scores"]["Meningioma"],
        "glioma": result["all_scores"]["Glioma"],
    })

with open("predictions.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
```

### Generate JSON Report

```python
import json
from datetime import datetime
from MRI.load_model import load_brain_classifier, classify_image

model, device = load_brain_classifier()

report = {
    "timestamp": datetime.now().isoformat(),
    "model_accuracy": "97.36%",
    "predictions": []
}

for img_path in Path("./scans").glob("**/*.jpg"):
    result = classify_image(str(img_path), model, device)
    report["predictions"].append(result)

with open("report.json", "w") as f:
    json.dump(report, f, indent=2)
```

---

## Training / Fine-tuning

Start fresh:
```bash
python3 train.py
```

Resume if interrupted:
```bash
python3 train.py --resume latest
```

---

## Files

```
MRI/
├── README.md              ← Full documentation (training, deployment, etc.)
├── QUICKSTART.md          ← This file
├── load_model.py          ← Model loader + helper functions
└── weights/
    └── brain_tumor_classifier_best.pth   ← Download this (~100MB)
```

## Download Weights

```bash
cd MRI/weights
wget https://github.com/Hamza09Hamza/MRI-scans/releases/download/v1.0/brain_tumor_classifier_best.pth
```

---

## Performance per Hardware

| Device | Speed | Memory |
|--------|-------|--------|
| RTX 4090 | 5ms | 2GB |
| RTX 3050 | 20ms | 2GB |
| M1 Pro | 15ms | 2GB |
| CPU | 100ms | 1GB |

---

## API Reference

### `load_brain_classifier(weights_path=None, device=None)`
Load pre-trained ResNet50 classifier.
```python
model, device = load_brain_classifier()
```

### `classify_image(image_path, model, device)`
Classify a single brain MRI image.
```python
result = classify_image("scan.jpg", model, device)
# Returns: {image, prediction, confidence, all_scores}
```

### `preprocess_image(image_path, device)`
Load and preprocess image tensor.
```python
x = preprocess_image("scan.jpg", device)  # Returns: (1, 3, 128, 128)
```

### `get_model_info()`
Get model architecture details.
```python
info = get_model_info()
print(info["accuracy_percentage"])  # "97.36%"
```

---

## Troubleshooting

**Model weights not found:**
- Download from GitHub releases (see README.md)
- Or train your own: `python3 train.py`

**GPU out of memory:**
- Use CPU: `model, device = load_brain_classifier(device="cpu")`
- Works, just slower (~100ms vs 10ms)

**Wrong predictions:**
- Check image is 128×128 or larger
- Ensure image is RGB (not grayscale)
- Try different preprocessing if needed

---

## More Info

- 📖 **Full Guide:** [README.md](README.md)
- 🔧 **Training:** [train.py](../train.py)
- 🏗️ **Architecture:** [ARCHITECTURE.md](../ARCHITECTURE.md)

**Status:** ✓ Production Ready  
**Accuracy:** 97.36% (Hospital Data)  
**Version:** 1.0
