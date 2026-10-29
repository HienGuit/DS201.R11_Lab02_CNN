import torch
from torch import nn

# Định nghĩa Inception block
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, f1, 1), nn.ReLU())
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, f3r, 1), nn.ReLU(),
            nn.Conv2d(f3r, f3, 3, padding=1), nn.ReLU())
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, f5r, 1), nn.ReLU(),
            nn.Conv2d(f5r, f5, 5, padding=2), nn.ReLU())
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, fp, 1), nn.ReLU())


    def forward(self, x):
        # dim=1 là chiều channel
        return torch.cat([self.b1(x), self.b2(x),
                          self.b3(x), self.b4(x)], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.specs = [(3,16,16,24,4,8,8), (56,32,32,48,8,16,16),
                      (112,64,32,64,16,32,32)]
        self.block1 = InceptionBlock(*self.specs[0])
        self.pool1 = nn.MaxPool2d(2,2)
        self.block2 = InceptionBlock(*self.specs[1])
        self.pool2 = nn.MaxPool2d(2,2)
        self.block3 = InceptionBlock(*self.specs[2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192,num_classes)
        self.block_channels = [56,112,192]

    def forward(self,x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        return self.fc(self.gap(x).flatten(1))
