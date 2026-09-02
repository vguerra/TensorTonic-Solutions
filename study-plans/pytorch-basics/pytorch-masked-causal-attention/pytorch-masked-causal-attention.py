import torch

def causal_attention(Q, K, V):
    """
    Returns: masked attention output tensor
    """
    d_k = Q.size(-1)
    attn_scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    mask = torch.triu(torch.ones_like(attn_scores), diagonal=1) == 1
    # print(mask)
    attn_scores.masked_fill_(mask, float("-inf"))
    attn_weights = torch.softmax(attn_scores, dim = -1)
    return attn_weights @ V