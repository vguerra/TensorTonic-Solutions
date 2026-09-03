import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        Returns: None
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Returns: output tensor
        """
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        return torch.relu(x + out)