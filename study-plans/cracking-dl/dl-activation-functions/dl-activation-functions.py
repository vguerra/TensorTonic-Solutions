import numpy as np

def activation_functions(x, activation):
    """
    Returns: list
    """
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    x = float(x)
    out = None
    grad = None
    match activation:
        case "relu":
            out = x if x > 0. else 0.
            grad = 1. if x > 0. else 0.
        case "sigmoid":
            out = sigmoid(x)
            grad = out * (1 - out)
        case "tanh":
            out = tanh(x)
            grad = 1 - out**2
        case "leaky_relu":
            out = x if x > 0 else 0.01*x
            grad = 1. if x > 0 else 0.01
        case "gelu":
            c = np.sqrt(2./math.pi)
            u = c * (x + 0.044715*x**3)
            t = tanh(u)
            out = 0.5 * x * (1 + t)
            grad = 0.5*(1 + t) + 0.5*x*(1 - t**2)*c*(1 + 3.*0.044715*x**2)
        case "swish":
            out = x * sigmoid(x)
            grad = sigmoid(x) + out *(1 - sigmoid(x))

    return round(out, 4) , round(grad, 4)
