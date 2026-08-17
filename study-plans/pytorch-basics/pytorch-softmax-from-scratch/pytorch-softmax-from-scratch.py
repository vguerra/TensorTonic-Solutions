import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    l = torch.tensor(logits, dtype=torch.float32)
    maxs, _ = torch.max(l, dim=-1, keepdims=True)
    exp_logits = torch.exp(l - maxs)
    return exp_logits/torch.sum(exp_logits, dim=-1, keepdims=True)
