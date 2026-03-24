import torch
import torch.nn as nn
from torchvision import models

class MultiOutputResNet18(nn.Module):
    def __init__(self, num_names=3, num_types=3, num_sleeves=4):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        in_features = backbone.fc.in_features

        self.head_name = nn.Linear(in_features, num_names)
        self.head_type = nn.Linear(in_features, num_types)
        self.head_sleeve = nn.Linear(in_features, num_sleeves)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.head_name(x), self.head_type(x), self.head_sleeve(x)
