import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Returns: None
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """
        Returns: convolved output tensor of shape (batch, out_channels, H-k+1, W-k+1)
        """
        batch_size = x.size(0)
        H_out = x.size(-2) - self.kernel_size + 1
        W_out = x.size(-1) - self.kernel_size + 1
        out = torch.empty(x.size(0), self.weight.size(0), H_out, W_out)

        flat_weights_transposed = self.weight.view(self.weight.size(0), -1).transpose(1, 0)
        print(self.weight.shape)
        print(flat_weights_transposed.shape)

        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size].flatten(start_dim=1)
                out[:, :, i, j] = patch @ flat_weights_transposed + self.bias
        return out
                
