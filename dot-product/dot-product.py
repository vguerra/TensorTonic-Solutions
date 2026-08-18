import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    x_arr = np.array(x)
    y_arr = np.array(y)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x array shape {x_arr.shape} != y array shape {y_arr.shape}")
    return np.dot(x_arr, y_arr)
