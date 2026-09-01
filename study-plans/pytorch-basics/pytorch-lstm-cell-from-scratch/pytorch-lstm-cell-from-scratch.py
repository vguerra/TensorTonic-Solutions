import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Returns: None
        """
        super().__init__()
        # each gate has two matrices: 
        # (h, d) for input
        # (h, h) for hidden state
        
        # forget gate
        self.W_if = nn.Parameter(torch.empty(hidden_size, input_size))
        torch.nn.init.xavier_uniform_(self.W_if)
        self.b_if = nn.Parameter(torch.zeros(hidden_size))
        self.W_hf = nn.Parameter(torch.empty(hidden_size, hidden_size))
        torch.nn.init.xavier_uniform_(self.W_hf)        
        self.b_hf = nn.Parameter(torch.zeros(hidden_size))

        # # input gate
        self.W_ii = nn.Parameter(torch.empty(hidden_size, input_size))
        torch.nn.init.xavier_uniform_(self.W_ii)
        self.b_ii = nn.Parameter(torch.zeros(hidden_size))
        self.W_hi = nn.Parameter(torch.empty(hidden_size, hidden_size))
        torch.nn.init.xavier_uniform_(self.W_hi)        
        self.b_hi = nn.Parameter(torch.zeros(hidden_size))

        # # candidate cell value
        self.W_ig = nn.Parameter(torch.empty(hidden_size, input_size))
        torch.nn.init.xavier_uniform_(self.W_ig)
        self.b_ig = nn.Parameter(torch.zeros(hidden_size))
        self.W_hg = nn.Parameter(torch.empty(hidden_size, hidden_size))
        torch.nn.init.xavier_uniform_(self.W_hg)        
        self.b_hg = nn.Parameter(torch.zeros(hidden_size))

        # # output gate
        self.W_io = nn.Parameter(torch.empty(hidden_size, input_size))
        torch.nn.init.xavier_uniform_(self.W_io)
        self.b_io = nn.Parameter(torch.zeros(hidden_size))
        self.W_ho = nn.Parameter(torch.empty(hidden_size, hidden_size))
        torch.nn.init.xavier_uniform_(self.W_ho)        
        self.b_ho = nn.Parameter(torch.zeros(hidden_size))


    def forward(self, x, h_prev, c_prev):
        """
        Returns: tuple of (h_t, c_t) tensors
        """
        f_t = torch.sigmoid(x @ self.W_if.transpose(-2,-1) + self.b_if +
                            h_prev @ self.W_hf.transpose(-2, -1) + self.b_hf)
        i_t = torch.sigmoid(x @ self.W_ii.transpose(-2,-1) + self.b_ii +
                            h_prev @ self.W_hi.transpose(-2, -1) + self.b_hi)
        g_t = torch.tanh(x @ self.W_ig.transpose(-2,-1) + self.b_ig +
                            h_prev @ self.W_hg.transpose(-2, -1) + self.b_hg)
        o_t = torch.sigmoid(x @ self.W_io.transpose(-2,-1) + self.b_io +
                            h_prev @ self.W_ho.transpose(-2, -1) + self.b_ho)

        c_t = torch.mul(f_t, c_prev) + torch.mul(i_t, g_t)
        h_t = torch.mul(o_t, torch.tanh(c_t))

        return h_t, c_t
        return torch.zeros_like(h_prev), torch.zeros_like(c_prev)        

