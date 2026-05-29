"""Chest X-ray analyzer - classification and disease detection."""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Any

from .base_analyzer import BaseAnalyzer


# CheXpert disease labels (14-class)
CHEXPERT_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Fracture",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
]


class ChestXrayAnalyzer(BaseAnalyzer):
    """Analyze chest X-rays for disease detection."""

    def __init__(self, model_checkpoint: str = None, device: str = "cpu"):
        super().__init__(model_checkpoint, device)
        self.num_classes = len(CHEXPERT_DISEASES)
        self.image_size = 224
        self.load_model()

    def load_model(self):
        """Load ResNet50 with 14 disease heads."""
        self.model = models.resnet50(weights="DEFAULT")
        # Replace classification head
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),  # 14 diseases
        )

        if self.model_checkpoint and Path(self.model_checkpoint).exists():
            state = torch.load(self.model_checkpoint, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"✓ Loaded checkpoint: {self.model_checkpoint}")
        else:
            print(f"⚠️  No checkpoint found at {self.model_checkpoint}, using pretrained ImageNet weights only")

        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_path: str) -> torch.Tensor:
        """Load and preprocess chest X-ray."""
        img = Image.open(image_path).convert("L")  # Grayscale
        img = img.resize((self.image_size, self.image_size))

        # Normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, 224, 224)

        # Repeat to 3 channels for ResNet
        img_tensor = img_tensor.repeat(1, 3, 1, 1)

        return img_tensor.to(self.device)

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze chest X-ray and detect diseases.

        Args:
            image_path: Path to X-ray image (DICOM, JPG, PNG, etc.)

        Returns:
            Dict with findings, confidence, and recommendations
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Preprocess
        img_tensor = self.preprocess(image_path)

        # Inference
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits)  # Multi-label sigmoid (can have multiple diseases)

        # Extract findings
        findings = {}
        disease_scores = {}
        threshold = 0.5

        for disease, prob in zip(CHEXPERT_DISEASES, probs[0]):
            prob_val = float(prob.item())
            disease_scores[disease] = prob_val
            if prob_val > threshold:
                findings[disease] = {
                    "detected": True,
                    "confidence": prob_val,
                    "severity": self._estimate_severity(disease, prob_val),
                }

        # Normal if no significant findings
        is_normal = len(findings) == 0

        # Overall confidence (max abnormality score)
        confidence = max(disease_scores.values()) if disease_scores else 0.0

        return {
            "modality": "chest_xray",
            "subtype": "radiograph",
            "normal": is_normal,
            "findings": findings,
            "disease_scores": disease_scores,
            "confidence": confidence,
            "threshold": threshold,
            "num_findings": len(findings),
        }

    @staticmethod
    def _estimate_severity(disease: str, confidence: float) -> str:
        """Estimate severity based on disease type and confidence."""
        if confidence < 0.6:
            return "mild"
        elif confidence < 0.8:
            return "moderate"
        else:
            return "severe"

    def get_modality_name(self) -> str:
        return "Chest X-ray"
