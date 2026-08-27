import torch
import torch.nn as nn

class CustomSGD(torch.optim.Optimizer):
    """
    Returns: loss or None from step()
    """

    def __init__(self, params, lr=0.01, momentum=0.0):
        defaults = dict(lr=lr, momentum_coef=momentum)
        super(CustomSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                momentum_coef = group['momentum_coef']
                for param in group["params"]:
                    if param.grad is not None:
                        state = self.state[param]
                        if len(state) == 0:
                            state['step'] = 0
                            state['momentum'] = torch.zeros_like(param, memory_format=torch.preserve_format)
    
                        momentum = state['momentum']
                        state['step'] += 1
    
                        momentum.mul_(momentum_coef).add_(param.grad)
                        param.sub_(lr * momentum)
                    
                
                
        
