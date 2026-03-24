import torch
import torch.nn as nn
from torchvision import models


class MultiOutputEfficientNetB0(nn.Module):
    def __init__(self, num_names=3, num_types=3, num_sleeves=4):
        super().__init__()

        backbone = models.efficientnet_b0(weights=None)

        # EfficientNet feature extractor
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        in_features = backbone.classifier[1].in_features

        # Heads (CONSISTENT NAMING)
        self.name_head = nn.Linear(in_features, num_names)
        self.type_head = nn.Linear(in_features, num_types)
        self.sleeve_head = nn.Linear(in_features, num_sleeves)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        out_name = self.name_head(x)
        out_type = self.type_head(x)
        out_sleeve = self.sleeve_head(x)

        return out_name, out_type, out_sleeve