import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A_arr = np.array(A)
    m, n = A_arr.shape
    T = np.zeros((n, m))
    print(A_arr.shape, T.shape)
    for r in range(m):
        T[:,r] = A_arr[r,:]

    return T
