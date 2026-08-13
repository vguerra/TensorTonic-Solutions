import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            masked = mask * x
            if self.p < 1:
                masked = masked / (1. - self.p)
            return masked
        return x
