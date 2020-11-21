"""
"""
import torch.nn as nn
import torch.nn.functional as F


class FacialModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=6):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, 5, 1, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 7, 2, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*4*4, 256)
        self.fc2 = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x