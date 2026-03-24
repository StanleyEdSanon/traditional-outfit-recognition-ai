import torch
import torch.nn as nn
import timm

class MultiOutputDeiT(nn.Module):
    def __init__(self, num_names=3, num_types=3, num_sleeves=4):
        super().__init__()
        
        # Load DeiT-Small/16 pretrained
        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True
        )
        
        # Remove original classifier
        in_features = self.backbone.head.in_features
        self.backbone.reset_classifier(0)
        
        # Multi-output heads
        self.name_head = nn.Linear(in_features, num_names)
        self.type_head = nn.Linear(in_features, num_types)
        self.sleeve_head = nn.Linear(in_features, num_sleeves)

    def forward(self, x):
        feats = self.backbone(x)
        return (
            self.name_head(feats),
            self.type_head(feats),
            self.sleeve_head(feats)
        )
MultiOutputDeiTSmall = MultiOutputDeiT
