import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    X_arr = torch.tensor(X, dtype=torch.float32)
    g = torch.tensor(gamma, dtype=torch.float32)
    b = torch.tensor(beta, dtype=torch.float32)


    mean = torch.mean(X_arr, dim=0, keepdim=True)
    res = X_arr - mean
    var = torch.mean(res ** 2, dim=0, keepdim=True)

    X_norm = res / torch.sqrt(var + eps)

    return (g * X_norm + b)
    
