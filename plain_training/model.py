import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class StandardMobileNetV2(nn.Module):
    """
    Standard Baseline Model:
    Takes an image and classifies it into one of 10 CIFAR-10 classes.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 1. Feature Extractor (Backbone):
        # Pre-trained on ImageNet. It has already learned to detect edges, shapes, and patterns.
        try:
            base = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except Exception:
            base = mobilenet_v2(pretrained=True)
            
        self.backbone = base.features               # Outputs (Batch, 1280, 7, 7)

        # 2. Global Average Pooling:
        # Compresses the (7, 7) spatial area down to (1, 1) per channel.
        self.pool = nn.AdaptiveAvgPool2d(1)         # Outputs (Batch, 1280, 1, 1)

        # 3. Classifier Head (Decision Maker):
        # Maps the 1280 extracted features to the final 10 class predictions.
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),                        # Randomly turns off 20% of neurons to prevent overfitting
            nn.Linear(1280, num_classes)            # 1280 features -> 10 class scores
        )

    def forward(self, x):
        # Pass 1: Extract features from image
        x = self.backbone(x)        # Shape: (Batch, 1280, 7, 7)
        
        # Pass 2: Average pool spatial dimensions
        x = self.pool(x)            # Shape: (Batch, 1280, 1, 1)
        
        # Pass 3: Flatten into a 1D vector of 1280 numbers
        x = x.flatten(1)            # Shape: (Batch, 1280)
        
        # Pass 4: Classifier outputs 10 logits
        out = self.classifier(x)    # Shape: (Batch, 10)
        
        return out

