import numpy as np
from sim.Util import clog2

def check_ATPP(bs_mat, Bx):
    n, N = bs_mat.shape
    log2_N = clog2(N)
    correct = np.zeros((n, ))
    for i in range(1, log2_N):
        correct += (2 ** (-i)) * Bx[:, log2_N-i]
        prec = np.mean(bs_mat[:, :(2**i)], axis=1)
        assert np.all(prec == correct)
