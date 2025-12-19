import torch
import torch.nn as nn
from torchvision import models

class SimpleCNN(nn.Module):
    """
    Простая сверточная сеть.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_resnet18(num_classes: int = 10):
    """
    Возвращает ResNet18 с замененной последней классификационной головой.
    """
    resnet = models.resnet18()
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet
