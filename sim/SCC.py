import numpy as np
from sim.Util import clog2
from sim.SNG import lfsr_sng

def scc(bsx, bsy):
    """Compute the stochastic cross-correlation between two bitstreams according to Eq. (1)
    in [A. Alaghi and J. P. Hayes, Exploiting correlation in stochastic circuit design]"""
    if bsx.dtype == np.dtype('uint8'):
        bsx = np.unpackbits(bsx)

    if bsy.dtype == np.dtype('uint8'):
        bsy = np.unpackbits(bsy)

    px = np.mean(bsx)
    py = np.mean(bsy)
    if px in (0, 1) or py in (0, 1):
        #raise ValueError("SCC is undefined for bitstreams with value 0 or 1") 
        return 1
    p_uncorr  = px * py
    p_actual  = np.mean(np.bitwise_and(bsx, bsy))
    if p_actual > p_uncorr:
        return (p_actual - p_uncorr) / (np.minimum(px, py) - p_uncorr)
    else:
        return (p_actual - p_uncorr) / (p_uncorr - np.maximum(px + py - 1, 0))
    
def scc_mat(bs_mat):
    n, _ = bs_mat.shape
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i,j] = scc(bs_mat[i, :], bs_mat[j, :])
    return C 
    
def reco_2(bsx, bsy):
    #Re-generate positive correlation between bitstreams

    if bsx.dtype == np.dtype('uint8'):
        bsx = np.unpackbits(bsx)

    if bsy.dtype == np.dtype('uint8'):
        bsy = np.unpackbits(bsy)

    px = np.mean(bsx)
    py = np.mean(bsy)

    N = bsx.size
    w = clog2(N)
    bs_mat = lfsr_sng(np.array([px, py]), N, w, corr=True)
    return bs_mat[0, :], bs_mat[1, :]