# model_def.py
import torch.nn as nn

class MiniVGG(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(3, 32, kernel_size=3, padding= 1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(kernel_size=2))
        layers.append(nn.Dropout(p=0.3))

        layers.append(nn.Conv2d(32, 64 , kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace= True))
        layers.append(nn.MaxPool2d(kernel_size=2))
        layers.append(nn.Dropout(p=0.25))
        layers.append(nn.Flatten(1))
        layers.append(nn.LazyLinear(512))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.LazyLinear(num_classes))
        self.block = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x
