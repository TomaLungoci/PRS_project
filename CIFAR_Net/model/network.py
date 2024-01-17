import torch
from model.kernel import ConvModule
import torch.nn as nn

class CIFAR_Net(torch.nn.Module):
    def __init__(self, module, in_channels, dim_encoder):
        super(CIFAR_Net, self).__init__()
        self.module = module
        self.in_channels = in_channels
        self.dim_encoder = dim_encoder
        self.layers = self._create_layers(dim_encoder)
        self.fc1 = nn.Linear(dim_encoder[-1] * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, self.dim_encoder[-1] * 4 * 4)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def _create_layers(self, dim_encoder):
        layers = []
        for channels in dim_encoder:
            layers.append(self.module(self.in_channels, channels))
            self.in_channels = channels
        return nn.Sequential(*layers)
    


class DINOv2(nn.Module):
    def __init__(self):
        super(DINOv2, self).__init__()
        self.backbone = dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.fc = nn.Linear(384, 64)
        self.head = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.backbone.norm(x)
        x = self.fc(x)
        return self.head(x)