import numpy as np
from sim.Util import *

B_mat_dict = {}
def B_mat(n, lsb='left'):
    """Generate a 2^n by n matrix of all of the possible values of n bits"""
    if f"{n}-{lsb}" in B_mat_dict.keys():
        return B_mat_dict[n]
    B = np.zeros((2 ** n, n), dtype=bool)
    for i in range(2 ** n):
        B[i][:] = bin_array(i, n, lsb=lsb)
    B_mat_dict[n] = B
    return B

def get_actual_vin(bs_mat, lag=0):
    if len(bs_mat.shape) == 1:
        bs_mat = np.expand_dims(bs_mat, axis=0)
    n, N = bs_mat.shape

    if lag > 0: #for autocorrelation analysis
        bs_mat_new = np.zeros(((lag+1)*n, N), dtype=np.bool_)
        for i in range(n):
            row = bs_mat[i, :]
            bs_mat_new[i, :] = row
            for j in range(lag):
                bs_mat_new[i+n, :] = np.roll(row, j+1) #FIXME: need to shift instead of roll
        bs_mat = bs_mat_new
        n = (lag+1)*n

    Vin = np.zeros(2 ** n)
    uniques, counts = np.unique(bs_mat.T, axis=0, return_counts=True)
    for unq, cnt in zip(uniques, counts):
        Vin[bit_vec_to_int(unq)] = cnt / N
    return Vin

def sample_from_ptv(ptv, N):
    """Uniformly sample N bitstream samples (each of width n) from a given PTV"""
    n = int(np.log2(ptv.shape[0]))
    bs_mat = np.zeros((n, N), dtype=np.uint8)
    for i in range(N):
        sel = np.random.choice(ptv.shape[0], p=ptv)
        bs_mat[:, i] = bin_array(sel, n)
    return bs_mat

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

def apply_ptm_to_bs(bs_mat, Mf):
    """Given a set of input bitstrems, compute the output bitstreams for the circuit defined by the PTM Mf"""
    n, N = bs_mat.shape
    n2, k2 = Mf.shape
    k = np.log2(k2).astype(np.int32)
    ints = int_array(bs_mat.T, lsb='right')
    bs_out = np.zeros((k, N), dtype=np.bool_)
    bm = B_mat(k, lsb='right')
    for i in range(N):
        bs_out[:, i] = Mf[ints[i], :] @ bm
    
    return bs_out

def reduce_func_mat(Mf, idx, p):
    """Reduce a PTM matrix with a known probability value on one input"""
    n, k = np.log2(Mf.shape).astype(np.uint16)
    ss1, ss2 = [], []
    for i in range(2 ** n):
        if i % (2 ** (idx + 1)) < 2 ** idx:
            ss1.append(i)
        else:
            ss2.append(i)
    Mff = Mf.astype(np.float32)
    return Mff[ss1, :] * p + Mff[ss2, :] * (1-p)

def TT_to_ptm(TT, n, k, lsb='right'):
    Mf = np.zeros((2**n, 2**k), dtype=np.bool_)
    for i in range(2**n):
        x = int_array(TT[i, :].reshape(1, k), lsb=lsb)[0]
        Mf[i, x] = True
    return Mf

#SEM generation
def get_SEMs_from_ptm(Mf, k, nc, nv):
    T = Mf @ B_mat(k, lsb='right') #2**(nc+nv) x k
    Fs = []
    for i in range(k):
        Fs.append(T[:, i].reshape(2**nv, 2**nc))
    return Fs

def get_weight_matrix_from_ptm(Mf, k, nc, nv):
    Fs = get_SEMs_from_ptm(Mf, k, nc, nv)
    W = np.empty((2**nv, k))
    for i in range(k):
        W[:, i] = np.sum(Fs[i], axis=1)
    return W / 2**nc