"""Base class for all modality-specific analyzers."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path


class BaseAnalyzer(ABC):
    """Abstract base class for radiology modality analyzers."""

    def __init__(self, model_checkpoint: str = None, device: str = "cpu"):
        """
        Initialize analyzer.

        Args:
            model_checkpoint: Path to trained model weights
            device: torch device (cpu, cuda, mps)
        """
        self.model_checkpoint = model_checkpoint
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load model weights from checkpoint."""
        pass

    @abstractmethod
    def preprocess(self, image_path: str) -> Any:
        """
        Preprocess image for model input.

        Args:
            image_path: Path to DICOM, NIfTI, or image file

        Returns:
            Preprocessed tensor ready for inference
        """
        pass

    @abstractmethod
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image and return findings.

        Args:
            image_path: Path to medical image

        Returns:
            Dict with:
                - modality: str (e.g., "chest_xray")
                - findings: Dict of detected pathologies/structures
                - confidence: float (0-1)
                - metadata: Dict of image-specific info
        """
        pass

    def get_modality_name(self) -> str:
        """Return human-readable modality name."""
        return self.__class__.__name__

    def is_ready(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self.model is not None
