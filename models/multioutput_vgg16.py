import torch
import torch.nn as nn
from torchvision import models

class MultiOutputVGG16(nn.Module):
    def __init__(self, num_names=3, num_types=3, num_sleeves=4):
        super().__init__()

        backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Feature extractor
        self.features = backbone.features

        # VGG16 flatten size after adaptive pooling
        self.avgpool = backbone.avgpool

        # IMPORTANT:
        # checkpoint expects:
        # fc.0 : Linear(25088 -> 4096)
        # fc.3 : Linear(4096 -> 1024)
        self.fc = nn.Sequential(
            nn.Linear(25088, 4096),   # fc.0
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),    # fc.3  <-- this is the critical fix
            nn.ReLU(True),
            nn.Dropout(0.5),
        )

        self.head_name = nn.Linear(1024, num_names)
        self.head_type = nn.Linear(1024, num_types)
        self.head_sleeve = nn.Linear(1024, num_sleeves)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        out_name = self.head_name(x)
        out_type = self.head_type(x)
        out_sleeve = self.head_sleeve(x)

        return out_name, out_type, out_sleeve