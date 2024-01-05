import numpy as np
from sim.Util import bin_array

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

def get_vin_mc1(Pin):
    """Generates a Vin vector for bitstreams mutually correlated with ZSCC=1"""
    n = Pin.size
    Vin = np.zeros(2 ** n)
    Vin[0] = 1 - np.max(Pin)
    Vin[2 ** n - 1] = np.min(Pin)
    Pin_sorted = np.argsort(Pin)[::-1]
    i = 0
    for k in range(1, n):
        i += 2 ** Pin_sorted[k - 1]
        Vin[i] = Pin[Pin_sorted[k - 1]] - Pin[Pin_sorted[k]]
    return np.round(Vin, 12)

def get_vin_mc0(Pin):
    """Generates a Vin vector for bitstreams mutually correlated with ZSCC=0"""
    n = Pin.size
    Bn = B_mat(n)
    return np.prod(Bn * Pin + (1 - Bn) * (1 - Pin), 1)

def get_vin_mcn1(Pin):
    """Generates a Vin vector for bitstreams mutually correlated with ZSCC=-1"""
    if np.sum(Pin) > 1:
        return None
    n = Pin.size
    Vin = np.zeros(2 ** n)
    Vin[0] = 1 - np.sum(Pin)
    Pin_sorted = np.argsort(Pin)[::-1]
    for k in range(n):
        i = 2 ** Pin_sorted[k]
        Vin[i] = Pin[Pin_sorted[k]]
    return np.round(Vin, 12)

#SEM generation
def get_SEMs_from_ptm(Mf, k, nc, nv):
    T = Mf @ B_mat(k) #2**(nc+nv) x k
    Fs = []
    for i in range(k):
        Fs.append(T[:, i].reshape(2**nc, 2**nv).T)
    return Fs

def get_weight_matrix_from_ptm(Mf, k, nc, nv):
    Fs = get_SEMs_from_ptm(Mf, k, nc, nv)
    W = np.empty(2**nv, k)
    for i in range(k):
        W[:, i] = np.sum(Fs[i], axis=0)
    return W