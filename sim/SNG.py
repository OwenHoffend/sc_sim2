import numpy as np

def sng(parr, N, w, c, rns, pcc):
    n = parr.size

    #Convert parr into binary array
    parr_bin = np.zeros((n, w), dtype=np.bool_)
    for i in range(n):
        cmp = 0.5
        p = parr[i]
        for j in reversed(range(w)):
            if p >= cmp:
                parr_bin[i, j] = True
                p -= cmp
            else:
                parr_bin[i, j] = False
            cmp /= 2
    
    #Generate the random bits
    bs_mat = np.zeros((n, N), dtype=np.bool_)
    r = rns(w, N)
    for i in range(n):
        p = parr_bin[i, :]
        for j in range(N):
            bs_mat[i, j] = pcc(r[:, j], p)
        if not c: #if not correlated, get a new independent rns sequence
            r = rns(w, N)

    return np.packbits(bs_mat, axis=1)

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

#Generate a bitstream with maximum possible streaming accuracy
def streaming_accurate_SNG(px, N):
    rsum = 0
    bs = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        et_err0 = np.abs(rsum/(i+1)-px)
        et_err1 = np.abs((rsum+1)/(i+1)-px)
        if et_err1 < et_err0:
            rsum += 1
            bs[i] = True
        else:
            bs[i] = False
    return np.packbits(bs)