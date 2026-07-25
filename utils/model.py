import torch
import torch.nn as nn
from torchvision import models


class TumorCNN(nn.Module):
    """
    Research classifier built from an ImageNet-pretrained ResNet50 backbone
    and a regularized multi-layer classification head.

    Dropout and batch normalization are implementation choices; they do not
    by themselves establish clinical robustness or generalization.
    """

    def __init__(self, config):
        super().__init__()
        num_classes = config["data"]["num_classes"]
        dropout = config["training"]["dropout_rate"]

        # ResNet50 backbone pretrained on ImageNet.
        self.backbone = models.resnet50(weights="DEFAULT")

        # Replace the original ImageNet classifier with a feature extractor.
        num_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()

        # Regularized research classifier head.
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)  # (batch, 2048)
        output = self.classifier(features)
        return output

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
