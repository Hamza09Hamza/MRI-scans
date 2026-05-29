"""Brain MRI analyzer - tumor segmentation and classification."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from utils.unet_3d import UNet3D


class BrainMRIAnalyzer:
    """Analyze brain MRI for tumor segmentation and classification."""

    TUMOR_CLASSES = {
        0: "background",
        1: "necrotic_core",
        2: "edema",
        3: "enhancing_tumor",
    }

    def __init__(self, model_checkpoint: str = None, device: str = "cpu"):
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.model = None
        self.load_model()

    def load_model(self):
        """Load 3D U-Net for tumor segmentation."""
        self.model = UNet3D(in_channels=2, out_channels=4, features=32)

        if self.model_checkpoint and Path(self.model_checkpoint).exists():
            state = torch.load(self.model_checkpoint, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"✓ Loaded checkpoint: {self.model_checkpoint}")
        else:
            print(f"⚠️  No checkpoint found, using untrained model")

        self.model = self.model.to(self.device)
        self.model.eval()

    def analyze(self, volume_path: str) -> Dict[str, Any]:
        """
        Analyze brain MRI volume and segment tumor.

        Args:
            volume_path: Path to NIfTI file or DICOM series

        Returns:
            Dict with segmentation mask, statistics, and classification
        """
        raise NotImplementedError(
            "Brain MRI analysis requires DICOM/NIfTI loading. "
            "Use existing train_3d_segmentation.py for full pipeline."
        )

    def get_modality_name(self) -> str:
        return "Brain MRI"
