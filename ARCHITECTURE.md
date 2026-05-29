# Universal Radiology AI Platform - Architecture & Implementation Plan

## Executive Summary

Transform the brain tumor segmentation pipeline into a scalable multi-modality radiology AI platform that replaces ChatGPT API calls with specialized, locally-run models.

**Current State:** Brain tumor classification + 3D segmentation (MRI-only)  
**Target State:** Multi-modality analysis (Chest X-ray, CT, Ultrasound, Mammography, Spine X-ray, etc.) with automated report generation

---

## Phase 1: Architecture Design (Week 1)

### 1.1 Modular System Design

```
radiology_ai/
├── core/
│   ├── base_analyzer.py        # Abstract base for all modalities
│   ├── report_generator.py      # Unified report generation engine
│   └── ensemble.py              # Multi-model orchestration
│
├── modalities/
│   ├── chest_xray/
│   │   ├── classifier.py        # Normal/Abnormal/Disease classification
│   │   ├── models/
│   │   └── config.yaml
│   ├── ct_scan/
│   │   ├── segmentation.py      # 3D organ/pathology segmentation
│   │   ├── models/
│   │   └── config.yaml
│   ├── brain_mri/               # Existing brain tumor system
│   ├── mammography/
│   └── ultrasound/
│
├── data_loaders/
│   ├── dicom_loader.py          # DICOM handling
│   ├── nifti_loader.py          # NIfTI (existing)
│   └── format_converter.py      # Auto-detect file format
│
├── inference/
│   ├── api_server.py            # FastAPI REST endpoints
│   ├── batch_processor.py       # Batch inference on scans
│   └── cache_manager.py         # Model caching
│
└── datasets/
    ├── chest_xray/
    │   ├── download_chexpert.py
    │   ├── download_nih_cxr.py
    │   └── preprocessing.py
    ├── ct/
    │   └── download_luna16.py
    └── general/
        └── dataset_registry.py
```

### 1.2 Model Strategy: Hybrid Approach

**NOT** one giant model (too slow to train, hard to iterate)  
**NOT** separate models per type (maintenance nightmare)

**INSTEAD: Modular Specialists**

```
┌─────────────────────────────────────────────────────┐
│            Radiology AI Platform                    │
├─────────────────────────────────────────────────────┤
│                Input Router                         │
│  (auto-detect scan type via DICOM metadata)        │
├──────┬──────────┬──────────┬──────────┬──────────┐
│Chest │   CT     │  Brain   │ Mammo    │  Spine   │
│ X-ray│  Organs  │  Tumor   │          │          │
├──────┼──────────┼──────────┼──────────┼──────────┤
│ ResNet│ UNet3D  │ UNet3D   │ ResNet   │ ResNet   │
│ 152   │ (organ) │ (tumor)  │ 152      │ 152      │
├──────┼──────────┼──────────┼──────────┼──────────┤
│Report │Report   │Report    │Report    │Report    │
│Engine │Engine   │Engine    │Engine    │Engine    │
└──────┴──────────┴──────────┴──────────┴──────────┘
         ↓
    Unified JSON/PDF Report
```

---

## Phase 2: Data Strategy (Weeks 1-3)

### 2.1 Public Datasets (Free & Large-Scale)

| Modality | Dataset | Size | Link | License |
|----------|---------|------|------|---------|
| Chest X-ray | CheXpert | 224K images | https://stanfordmlgroup.github.io/competitions/chexpert/ | CC BY 4.0 |
| Chest X-ray | NIH Chest X-ray | 112K images | https://www.nih.nlm.nih.gov/research/bel/chest-xray-download.html | Public Domain |
| CT Lungs | LUNA16 | 1,186 CTs | https://luna16.grand-challenge.org/ | CC BY 4.0 |
| CT Organs | Medical Decathlon | Multi-organ | https://medicaldecathlon.com/ | CC BY 4.0 |
| Brain (existing) | BraTS2020 | 494 MRIs | Already downloaded | CC BY 4.0 |
| Ultrasound | BUSI (Breast) | 780 images | https://github.com/suansuansuansuansuansuan/Breast-Ultrasound-Dataset | Public Domain |

**Total: 440K+ images for training**

### 2.2 Data Folder Structure

```
data/
├── chest_xray/
│   ├── chexpert/
│   │   ├── raw/               # Downloaded
│   │   ├── processed/         # Resized, normalized
│   │   └── labels.csv
│   └── nih_cxr/
│       ├── images/
│       └── metadata.csv
├── ct/
│   ├── luna16/
│   │   ├── raw_dcm/
│   │   ├── nifti/
│   │   └── labels/
│   └── medical_decathlon/
├── brain_mri/
│   ├── BraTS2020/
│   └── upenn_gbm/
├── mammography/
└── ultrasound/
```

---

## Phase 3: Implementation (Weeks 2-6)

### 3.1 Step-by-Step Rollout

**Week 2: Chest X-ray Classification**
- Download CheXpert (224K labeled images)
- Binary classifier: Normal vs Abnormal
- Multi-class: 14 disease labels (pneumonia, pneumothorax, etc.)
- ResNet50 baseline → ~95% accuracy expected
- Basic report: "Findings: Pneumonia detected. Recommendation: Follow-up CT"

**Week 3: CT Organ Segmentation**
- Download LUNA16 + Medical Decathlon
- 3D UNet for lung segmentation
- Extend to multi-organ (liver, kidney, spleen)
- Report: "Segmented organs: Lungs (85% volume), Liver (normal), Kidney (cyst detected)"

**Week 4: Report Generation Engine**
- Template-based system (findings → templates → text)
- LLM augmentation (optional: use local model or API for narrative)
- PDF export with annotated images
- Structured JSON output (for downstream integration)

**Week 5: Brain Tumor (Keep & Extend)**
- Keep existing 3D U-Net system
- Add pre-op/post-op comparison
- Volumetric analysis: "Tumor volume 45.2 cm³ (-12% from baseline)"

**Week 6: API + Unified Interface**
- FastAPI REST service
- Single endpoint: POST scan → JSON report
- Batch processing: upload folder of scans
- Web UI (optional): drag-drop DICOM files

---

## Phase 4: Technical Details

### 4.1 Chest X-ray Implementation

```python
# modalities/chest_xray/classifier.py

class ChestXrayAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.model = models.resnet50(weights='DEFAULT')
        # Replace head: 2048 → 14 disease classes
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 14),  # 14 disease labels
        )
        self.load_checkpoint('models/chexpert_resnet50_best.pth')
    
    def analyze(self, image_path):
        """Analyze chest X-ray and return findings + confidence."""
        img = load_and_preprocess(image_path, size=224)
        with torch.no_grad():
            logits = self.model(img)
            probs = torch.softmax(logits, dim=1)
        
        findings = {}
        for label, prob in zip(DISEASE_LABELS, probs[0]):
            if prob > 0.5:  # confidence threshold
                findings[label] = float(prob)
        
        return {
            "modality": "chest_xray",
            "findings": findings,
            "normal": len(findings) == 0,
            "confidence": max(probs[0]).item(),
        }
```

### 4.2 Unified Report Generator

```python
# core/report_generator.py

class ReportGenerator:
    def __init__(self):
        self.templates = {
            "chest_xray": ChestXrayTemplate(),
            "ct": CTTemplate(),
            "brain_mri": BrainMRITemplate(),
        }
    
    def generate(self, analysis_result):
        """Convert analysis → human-readable report."""
        modality = analysis_result["modality"]
        template = self.templates[modality]
        
        # Extract findings
        findings = analysis_result["findings"]
        
        # Generate narrative
        narrative = template.generate_narrative(findings)
        
        # Generate recommendations
        recommendations = template.generate_recommendations(findings)
        
        # Structured report
        report = {
            "timestamp": datetime.now().isoformat(),
            "modality": modality,
            "summary": narrative,
            "findings": findings,
            "recommendations": recommendations,
            "confidence_score": analysis_result.get("confidence", 0),
        }
        
        return report
```

### 4.3 API Endpoint

```python
# inference/api_server.py

from fastapi import FastAPI, UploadFile, File
from enum import Enum

app = FastAPI(title="Radiology AI")

class ReportFormat(str, Enum):
    JSON = "json"
    PDF = "pdf"
    TEXT = "text"

@app.post("/analyze")
async def analyze_scan(file: UploadFile = File(...), format: ReportFormat = ReportFormat.JSON):
    """
    Analyze a radiological scan and generate a report.
    
    Args:
        file: DICOM, NIfTI, or common image format
        format: json, pdf, or text
    
    Returns:
        Structured report with findings and recommendations
    """
    # Save uploaded file
    path = save_upload(file)
    
    # Auto-detect modality from DICOM metadata
    modality = detect_modality(path)
    
    # Load appropriate analyzer
    analyzer = get_analyzer(modality)
    
    # Run analysis
    result = analyzer.analyze(path)
    
    # Generate report
    report = report_generator.generate(result)
    
    # Format output
    if format == ReportFormat.JSON:
        return report
    elif format == ReportFormat.PDF:
        return generate_pdf_report(report, path)
    else:
        return {"report": report_to_text(report)}

@app.post("/batch-analyze")
async def batch_analyze(folder: str):
    """Analyze all DICOM files in a folder."""
    results = []
    for file_path in Path(folder).glob("**/*.dcm"):
        result = analyze_scan(file_path)
        results.append(result)
    return {"count": len(results), "reports": results}
```

---

## Phase 5: Model Training Strategy

### 5.1 Transfer Learning (Fast Path)

```
Pre-trained backbone (ImageNet) → Fine-tune on medical dataset
├─ Chest X-ray: ResNet50 + 10K images = 2 hours training
├─ CT segmentation: UNet3D + LUNA16 = 48 hours
└─ Mammography: ResNet50 + subset = 1 hour
```

### 5.2 Training Infrastructure

- **GPU Requirements:** RTX 4090 (ideal) or 2x RTX 3090
- **Batch Processing:** 50-100 scans in parallel
- **Checkpoint System:** Auto-save best model per modality
- **Registry:** `models/registry.json` tracks all trained models

```json
{
  "chest_xray": {
    "model_name": "chexpert_resnet50",
    "version": "1.0",
    "accuracy": 0.95,
    "checkpoint": "models/chest_xray/best.pth",
    "training_date": "2024-05-03"
  },
  "ct_lungs": {
    "model_name": "luna16_unet3d",
    "version": "1.0",
    "dice_score": 0.92,
    "checkpoint": "models/ct/lungs_best.pth",
    "training_date": "2024-05-05"
  }
}
```

---

## Phase 6: Monitoring & Deployment

### 6.1 Metrics Dashboard

Track per-modality:
- Accuracy / Dice score
- Inference time per scan
- Confidence calibration
- Edge cases (when model is uncertain)

### 6.2 Deployment Options

**Option A: Docker Container**
```bash
docker build -t radiology-ai .
docker run -p 8000:8000 radiology-ai
# Access at http://localhost:8000
```

**Option B: On-Premises Server**
- GPU machine (RTX 4090 recommended)
- 500GB+ storage for models + cache
- FastAPI runs on 8000

**Option C: Hybrid Cloud**
- Lightweight scans → local (X-ray, mammography)
- Heavy scans → cloud (CT, 3D MRI)

---

## Timeline & Milestones

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Architecture + data plan | This document |
| 2 | Chest X-ray classifier | CheXpert model (95% acc) |
| 3 | CT segmentation | LUNA16 model (92% Dice) |
| 4 | Report engine | Template system + PDF export |
| 5 | Brain tumor integration | Keep existing, add compare feature |
| 6 | API + unified UI | FastAPI + web interface |
| 7 | Testing + hardening | Edge cases, error handling |
| 8 | Deployment | Docker, documentation |

---

## Resource Requirements

### Hardware (Production)
- GPU: RTX 4090 or 2x RTX 3090 (training)
- CPU: 16+ cores
- RAM: 128GB+ (for batch processing)
- Storage: 1TB (models + cache)

### Software
- PyTorch 2.0+
- FastAPI
- MONAI (medical imaging)
- Pydicom (DICOM handling)
- ReportLab (PDF generation)
- Redis (caching, optional)

### Team
- 1 ML Engineer (model training)
- 1 Backend Engineer (API + deployment)
- 1 QA Engineer (validation on real scans)

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Data imbalance (more CXR than CT) | Poor CT model performance | Oversample CT, use class weights |
| Model latency (too slow for clinic) | Not usable in real-time | Profile inference, use TorchScript/ONNX |
| False negatives (miss disease) | Patient harm | High recall threshold, double-check flagged cases |
| HIPAA compliance | Legal liability | Encrypt in transit/at rest, no patient identifiers |
| Model drift (accuracy drops over time) | Degraded service | Monitor metrics, retrain monthly |

---

## Success Criteria

✓ Chest X-ray: 95%+ accuracy on test set  
✓ CT lungs: 90%+ Dice score on segmentation  
✓ Report generation: <5 sec per scan  
✓ API: <100ms latency per request  
✓ Uptime: 99.9% availability  
✓ False negative rate: <2% (patient safety)

