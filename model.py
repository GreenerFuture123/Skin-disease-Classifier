"""Stage 4: MobileNetV2 with a frozen base and a new 4-class final layer.

Transfer learning: start from a network already trained on ImageNet
(1.2Million everyday photos) and only retrain the last layer for skin images.
With ~1,000 training images, training the whole network from scratch would
just memorise them. The pretrained layers already pick out edges, textures
and shapes, which is most of what's needed.

MobileNetV2 because it's small (~2.2M parameters) and trains quickly on a laptop.
"""

import torch.nn as nn
from torchvision import models

from data import CLASSES


def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze the pretrained base so training doesn't change it. (False)
    # (train.py unfreezes the last few blocks later, for fine-tuning.)
    for param in model.features.parameters():
        param.requires_grad = False

    # Swap the 1000-class ImageNet output layer for the 4-class one.
    # classifier = [Dropout, Linear] where index 1 is the Linear layer.
    in_features = model.classifier[1].in_features   # 1280
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    return model


if __name__ == "__main__":
    # Quick check before moving on: -trainable should be tiny (just the new layer, 1280*4 + 4 = 5,124)-
    model = build_model()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
