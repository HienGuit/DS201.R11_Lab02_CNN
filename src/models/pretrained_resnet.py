import torch
from torch import nn


class PretrainedResnet(nn.Module):
    def __init__(self, num_classes=21, freeze_backbone=False, pretrained=True,
                 model_id='microsoft/resnet-50'):
        super().__init__()
        from transformers import ResNetConfig, ResNetForImageClassification
        base = (ResNetForImageClassification.from_pretrained(model_id) if pretrained
                else ResNetForImageClassification(ResNetConfig()))
        self.resnet = base.resnet
        self.classifier = nn.Linear(2048,num_classes,bias=True)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False
            self.resnet.eval()

    def train(self,mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.resnet.eval()
        return self

    def forward(self,images):
        features = self.resnet(images).pooler_output
        features = features.squeeze(-1).squeeze(-1)
        return self.classifier(features)
