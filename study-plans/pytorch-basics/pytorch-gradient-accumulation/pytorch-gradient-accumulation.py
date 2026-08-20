import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns: tuple of (updated_weights_list, last_avg_gradient_list)
    """
    w = torch.tensor(w_init, dtype=torch.float32, requires_grad=True)

    for idx, data in enumerate(micro_batches):
        inputs, targets = torch.tensor(data[0], dtype=torch.float32), torch.tensor(data[1], dtype=torch.float32)
        l = (inputs @ w - targets) ** 2
        l.backward()
        if (idx + 1) % accum_steps == 0:
            with torch.no_grad():
                avg_grad = w.grad / accum_steps
                w.sub_(lr * avg_grad)
            w.grad.zero_()

    return w.tolist(), avg_grad.tolist()
    
