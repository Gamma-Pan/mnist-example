import torch
from torch import nn
from torch.nn import functional as F


# DEFINE THE MODULE
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2, ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 2, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        # fully connected layers, output 10 classes
        self.lin = nn.Sequential(
            nn.Linear(32 * 8 * 8, 800),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200,10),
        )

    def forward(self, x):
        x = self.batch_norm1(self.conv1(x))
        x = self.batch_norm2(self.conv2(x))
        # flatten input
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        return x
