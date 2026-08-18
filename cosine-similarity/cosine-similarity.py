import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a_arr = np.array(a)
    b_arr = np.array(b)

    a_norm = np.linalg.norm(a_arr)
    b_norm = np.linalg.norm(b_arr)

    div_term = a_norm * b_norm

    if div_term == 0.0:
        return 0.0

    return np.dot(a_arr, b_arr) / div_term