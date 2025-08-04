import torch
import torch.nn as nn
from torchvision import models

class MobileScreenClassifier(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(MobileScreenClassifier, self).__init__()
        
        # 加载预训练的ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # 冻结前面的层，只训练最后几层
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 解冻最后的几个残差块
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
            
        # 替换最后的全连接层
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_all(self):
        """解冻所有层用于fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

def create_model(num_classes=7, pretrained=True):
    model = MobileScreenClassifier(num_classes=num_classes, pretrained=pretrained)
    return model