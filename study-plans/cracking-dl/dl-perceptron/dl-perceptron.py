import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    # X has shape n, d
    # list of binary labels 0, 1 , length n

    X = np.array(X)
    Y = np.array(y)
    n, d = X.shape

    w = np.zeros((d))
    b = 0

    for step in range(epochs):
        for x, y in zip(X, Y):
            y_pred = 1.0 if np.dot(w, x) + b >= 0 else 0.
            error = y - y_pred
            w += lr * error * x
            b += lr * error

    return w.tolist(), b.item()


    