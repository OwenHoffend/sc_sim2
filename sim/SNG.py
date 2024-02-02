import numpy as np
from sim.Util import *
from sim.PCC import *
from sim.RNS import *

def sng(parr, N, w, rns, pcc, corr=0, pack=True):
    n = parr.size
    pbin = parr_bin(parr, w, lsb="right")
    
    #Generate the random bits
    bs_mat = np.zeros((n, N), dtype=np.bool_)
    r = rns(w, N)
    for i in range(n):
        p = pbin[i, :]
        for j in range(N):
            bs_mat[i, j] = pcc(r[:, j], p)
        if not corr: #if not correlated, get a new independent rns sequence
            r = rns(w, N)

    if pack:
        return np.packbits(bs_mat, axis=1)
    else:
        return bs_mat

def lfsr_sng(parr, N, w, **kwargs):
    return sng(parr, N, w, lfsr, CMP, **kwargs)

def van_der_corput_sng(parr, N, w, **kwargs):
    return sng(parr, N, w, van_der_corput, CMP, **kwargs)

def counter_sng(parr, N, w, **kwargs):
    return sng(parr, N, w, counter, CMP, **kwargs)

def true_rand_sng(parr, N, w, **kwargs):
    return sng(parr, N, w, true_rand, CMP, **kwargs)

def CAPE_sng(parr, N, w, pack=True):
    """Design from: 
    T. -H. Chen, P. Ting and J. P. Hayes, 
    "Achieving progressive precision in stochastic computing
    """
    n = parr.size #number of bitstreams
    pbin = parr_bin(parr, w, lsb="right")
    ctr_list = [bin_array(i, n * w)[::-1] for i in range(N)]
    ctrs = np.array(ctr_list)

    bs_mat = np.zeros((n, N), dtype=np.bool_)
    for i in range(n):
        p = pbin[i, :]
        idx = np.array([x * n + i for x in range(w)])
        for j in range(N):
            c = ctrs[j, :]
            r = c[idx]
            bs_mat[i, j] = WBG(r, p)
    if pack:
        return np.packbits(bs_mat, axis=1)
    else:
        return bs_mat
    
#Generate a bitstream with maximum possible streaming accuracy
def SA_sng(parr, N, w, pack=True):
    n = parr.size
    rsum = 0.0
    bs_mat = np.zeros((n, N), dtype=np.bool_)
    for i, px in enumerate(parr):
        for j in range(N):
            et_err0 = np.abs(rsum/(j+1.0)-px)
            et_err1 = np.abs((rsum+1)/(j+1.0)-px)
            if et_err1 < et_err0:
                rsum += 1.0
                bs_mat[i, j] = True
            else:
                bs_mat[i, j] = False
    if pack:
        return np.packbits(bs_mat, axis=1)
    else:
        return bs_mat

def sng_from_pointcloud(parr, S, pack=True):
    _, N = S.shape
    d = parr.size
    bs_mat = np.zeros((d, N), dtype=np.bool_)
    parr *= (N ** (1/d))
    for i in range(N):
        bs_mat[:, i] = S[:, i] < parr 
    if pack:
        return np.packbits(bs_mat, axis=1)
    else:
        return bs_mat