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