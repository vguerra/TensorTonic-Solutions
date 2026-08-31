import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Returns: None
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # self.W_q = nn.Linear(d_model, d_model, bias=False)
        # self.W_k = nn.Linear(d_model, d_model, bias=False)
        # self.W_v = nn.Linear(d_model, d_model, bias=False)
        # self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.W_q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_k = nn.Parameter(torch.randn(d_model, d_model))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model))
        self.W_o = nn.Parameter(torch.randn(d_model, d_model))

    
    def forward(self, Q, K, V):
        """
        Returns: output tensor
        """
        query = Q @ self.W_q # (batch, seq_len, d_model)
        key = K @ self.W_k # (batch, seq_len, d_model)
        value = V @ self.W_v # (batch, seq_len, d_model)

        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.size(0), query.size(1), self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.softmax(query @ key.transpose(-2, -1) / (self.d_k ** 0.5), dim=-1)
        attn_weights = attn_scores @ value

        out = attn_weights.transpose(1, 2).contiguous().view(attn_weights.size(0), -1, self.d_model)

        return out @ self.W_o

