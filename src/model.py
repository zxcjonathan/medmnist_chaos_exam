import torch
import torch.nn as nn
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 調整骨幹網路的第一層，以適應不同的輸入通道數
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
        # 凍結參數
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 替換最後的全連接層
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.backbone(x)