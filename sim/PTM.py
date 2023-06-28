import numpy as np

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(
        list(np.binary_repr(num).zfill(m))
    ).astype(bool)[::-1] #Changed to [::-1] here to enforce ordering globally (12/29/2021)

B_mat_dict = {}
def B_mat(n):
    """Generate a 2^n by n matrix of all of the possible values of n bits"""
    if n in B_mat_dict.keys():
        return B_mat_dict[n]
    B = np.zeros((2 ** n, n), dtype=bool)
    for i in range(2 ** n):
        B[i][:] = bin_array(i, n)
    B_mat_dict[n] = B
    return B

def get_func_mat(func, n, k, **kwargs):
    """Compute the PTM for a boolean function with n inputs and k outputs
        Does not handle probabilistic functions, only pure boolean functions"""
    Mf = np.zeros((2 ** n, 2 ** k), dtype=bool)

    if k == 1:
        for i in range(2 ** n):
            res = func(*list(bin_array(i, n)), **kwargs)
            num = res.astype(np.uint8)
            Mf[i][num] = 1
    else:
        for i in range(2 ** n):
            res = func(*list(bin_array(i, n)), **kwargs)
            num = 0
            for idx, j in enumerate(res):
                if j:
                    num += 1 << idx
            Mf[i][num] = 1
    return Mf