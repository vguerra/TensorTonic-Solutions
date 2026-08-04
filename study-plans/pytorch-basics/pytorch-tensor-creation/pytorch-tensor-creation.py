import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    match method:
        case "zeros":
            z = torch.zeros(*shape)
        case "ones":
            z = torch.ones(*shape)
        case "full":
            z = torch.full(shape, value)
    return z.tolist()