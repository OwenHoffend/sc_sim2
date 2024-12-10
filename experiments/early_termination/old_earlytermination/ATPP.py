import numpy as np
from sim.Util import clog2

def check_ATPP(bs_mat, Bx):
    n, N = bs_mat.shape
    log2_N = clog2(N)
    correct = np.zeros((n, ))
    for i in range(1, log2_N):
        correct += (2 ** (-i)) * Bx[:, i-1]
        prec = np.mean(bs_mat[:, :(2**i)], axis=1)
        print(prec)
        assert np.all(prec == correct)

def print_precision_points(bs_mat):
    n, N = bs_mat.shape
    log2_N = clog2(N)
    for i in range(1, log2_N+1):
        print(np.mean(bs_mat[:, :(2**i)], axis=1))