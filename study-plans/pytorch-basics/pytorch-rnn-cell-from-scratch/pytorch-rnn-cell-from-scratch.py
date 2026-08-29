import torch
import torch.nn as nn

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Returns: None
        """
        super().__init__()
        self.W_ih = nn.Parameter(torch.rand(hidden_size, input_size))
        self.b_ih = nn.Parameter(torch.zeros(hidden_size))
        self.W_hh = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.b_hh = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h_prev):
        """
        Returns: new hidden state tensor
        """
        # x (batch, input_size)
        # h_prev (batch_hidden_size)
        # return h (batch, hidden_state)

        return torch.tanh(
            x @ self.W_ih.transpose(-1, -2) + self.b_ih + h_prev @ self.W_hh.transpose(-1, -2) + self.b_hh)