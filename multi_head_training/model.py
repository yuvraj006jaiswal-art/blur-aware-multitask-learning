import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MultiTaskMobileNetV2(nn.Module):
    """
    Multi-Task Model:
    Shares ONE backbone (feature extractor) but has TWO heads:
      - Head 1: Predicts image class (10 classes for CIFAR-10)
      - Head 2: Predicts blur level (3 classes: clean, mild, heavy)
    """
    def __init__(self, num_classes=10, num_blur_levels=3):
        super().__init__()
        
        # 1. Shared Feature Extractor (Backbone):
        try:
            base = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except Exception:
            base = mobilenet_v2(pretrained=True)
            
        self.backbone = base.features               # Shared features (Batch, 1280, 7, 7)
        self.pool     = nn.AdaptiveAvgPool2d(1)     # Compresses to (Batch, 1280, 1, 1)

        # 2. HEAD 1: Object Classification (What object is in this image?)
        self.class_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)            # 1280 -> 10 classes
        )

        # 3. HEAD 2: Blur Estimation (How blurry is this image?)
        self.blur_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_blur_levels)        # 1280 -> 3 blur levels
        )

    def forward(self, x):
        # Step 1: Extract shared features
        x = self.backbone(x)        # (Batch, 1280, 7, 7)
        x = self.pool(x)            # (Batch, 1280, 1, 1)
        features = x.flatten(1)     # (Batch, 1280)

        # Step 2: Feed the SAME features into both heads
        class_out = self.class_head(features)  # (Batch, 10)
        blur_out  = self.blur_head(features)   # (Batch, 3)

        return class_out, blur_out

