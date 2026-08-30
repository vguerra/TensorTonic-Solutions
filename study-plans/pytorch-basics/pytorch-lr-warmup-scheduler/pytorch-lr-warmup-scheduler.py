import math

def warmup_cosine_schedule(base_lr, warmup_steps, total_steps):
    """
    Returns: list of learning rates
    """
    lrs = []

    for step in range(warmup_steps):
        lr = base_lr * (step + 1) / warmup_steps
        lrs.append(lr)

    decay_steps = total_steps - warmup_steps
    for step in range(decay_steps):
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * step / decay_steps))
        lrs.append(lr)
    
    return lrs