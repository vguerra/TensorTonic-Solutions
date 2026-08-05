import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    match op:
        case "add":
            res = torch.add(x, y)
        case "multiply":
            res = torch.multiply(x , y)
        case "matmul":
            res = torch.matmul(x, y)
        case "power":
            res = torch.pow(x, y)
        case "max":
            res = torch.max(x, y)
            
            
    
    return res.tolist()