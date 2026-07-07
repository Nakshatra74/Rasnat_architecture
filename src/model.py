"""
ResNet-50 transfer learning model: load ImageNet-pretrained backbone,
replace the final fully connected layer to match the target number of classes.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def build_model(num_classes, freeze_backbone=False, pretrained=True):
    """
    Args:
        num_classes: number of output classes for the new FC head.
        freeze_backbone: if True, only the new FC layer is trained
                          (feature-extraction mode). If False, the whole
                          network is fine-tuned end-to-end.
        pretrained: load ImageNet-1k pretrained weights.

    Returns: torch.nn.Module
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer (originally 2048 -> 1000)
    # with one sized for the domain-specific dataset. New layer is trainable
    # by default even when the backbone is frozen.
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model
