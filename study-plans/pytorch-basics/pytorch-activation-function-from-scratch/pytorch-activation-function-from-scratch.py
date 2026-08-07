import torch

def activate(x, method="relu"):
    """
    Returns: list (activated tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    match method:
        case "relu":
            res = torch.clamp(x, 0.0)
        case "sigmoid":
            res = 1. / (1 + torch.exp(-x))
        case "tanh":
            res = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
        case "leaky_relu":
            res = torch.where(x > 0, x, 0.01 * x)

    return res.tolist()