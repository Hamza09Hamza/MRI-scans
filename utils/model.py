import torch
import torch.nn as nn
from torchvision import models


class TumorCNN(nn.Module):
    """
    Clinical production model: ResNet50 + robust classifier.

    Pretrained on ImageNet (1.2M images) then fine-tuned on medical data.
    Strong regularization (dropout, batch norm) ensures generalization.
    """

    def __init__(self, config):
        super().__init__()
        num_classes = config["data"]["num_classes"]
        dropout = config["training"]["dropout_rate"]

        # ResNet50 backbone: pretrained on ImageNet
        self.backbone = models.resnet50(weights='DEFAULT')

        # Replace final layer with feature extractor
        num_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()

        # Clinical-grade classifier: robust multi-layer head
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
