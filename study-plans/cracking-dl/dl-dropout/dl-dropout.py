import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X = np.array(X, dtype=np.float32)
    mask = np.array(mask)
    if mode == "test" or drop_prob == 0:
        return X
    
    return np.round((X * mask)/(1 - drop_prob), decimals=4).tolist()