import torch

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    match method:
        case "mse":
            t = torch.tensor(pred, dtype=torch.float32)
            p = torch.tensor(target, dtype=torch.float32)
            return torch.mean((t - p) ** 2).item()
        case "cross_entropy":
            logits = torch.tensor(pred, dtype=torch.float32)
            idx = torch.tensor(target, dtype=torch.int32)
            m = torch.max(logits, dim=-1, keepdim=True).values
            l = m + torch.log(torch.sum(torch.exp(logits - m), dim=-1, keepdim=True)) - logits[torch.arange(logits.shape[0]), idx]
            return torch.mean(l).item()
        case "huber":
            t = torch.tensor(pred, dtype=torch.float32)
            p = torch.tensor(target, dtype=torch.float32)

            a = torch.abs(p - t)
            l = torch.where(a <= delta, 0.5 * a ** 2, delta * (a - 0.5 * delta))
            return torch.mean(l).item()
            
            
