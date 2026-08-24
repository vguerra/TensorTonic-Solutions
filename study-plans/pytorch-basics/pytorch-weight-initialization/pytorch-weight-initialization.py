import torch

def initialize_weights(fan_in, fan_out, method):
    """
    Returns: tensor of shape (fan_out, fan_in) with initialized weights
    """
    match method:
        case 'xavier_uniform':
            factor = math.sqrt(6. / (fan_in + fan_out))
            return torch.empty(fan_out, fan_in).uniform_(-factor, factor)
        case 'xavier_normal':
            factor = math.sqrt(2. / (fan_in + fan_out))
            return torch.empty(fan_out, fan_in).normal_(0., factor)
        case 'he_uniform':
            factor = math.sqrt(6. / fan_in)
            return torch.empty(fan_out, fan_in).uniform_(-factor, factor)
        case 'he_normal':
            factor = math.sqrt(2. / fan_in)
            return torch.empty(fan_out, fan_in).normal_(0., factor)

    raise ValueError(f"Unknown initialization method {method}")
