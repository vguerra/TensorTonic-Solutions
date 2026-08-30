import torch

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: attention output tensor
    """
    d_k = Q.size(-1)
    attn_scores = torch.softmax(Q @ K.transpose(-2, -1)/ (d_k ** 0.5), dim=-1)
    return attn_scores @ V
    