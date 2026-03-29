import torch
import torch.nn as nn
from torchvision.models import resnet152


class DeepfakeModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        backbone = resnet152(weights="DEFAULT" if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        feats = self.backbone(x)       # [B*T, 2048]
        feats = feats.view(b, t, -1)
        feats = feats.mean(dim=1)      # [B, 2048]

        return self.classifier(feats)