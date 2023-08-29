import numpy as np
from sim.PTM import get_func_mat, B_mat
from sim.Util import bin_array

def ith_minterm(*x, mt=0):
    n = len(x)
    mt_bin = bin_array(mt, n)
    out = True
    for i in range(n):
        if mt_bin[i]:
            out = np.bitwise_and(out, x[i])
        else:
            out = np.bitwise_and(out, np.bitwise_not(x[i]))
    return out

def COMAX(f, nc, nv, k):
    ptm = get_func_mat(f, nc + nv, k)
    TT = ptm @ B_mat(k) # 2**n x k
    W = np.zeros((2**nv, k), dtype=np.int32)
    for i in range(k):
        W[:, i] = np.sum(TT[:, i].reshape(2**nv, 2**nc), axis=1)

    def fopt(*x):
        xc = x[:nc]
        xv = x[nc:]
        out = np.zeros((k, ), dtype=np.bool_)

        #pre-compute all constant terms
        cs = np.zeros((2**nc, ), dtype=np.bool_)
        cs[0] = ith_minterm(*xc, mt=0)
        for j in range(1, 2**nc):
            cs[j] = np.bitwise_or(cs[j-1], ith_minterm(*xc, mt=j))

        for ell in range(k):
            for i in range(2**nv):
                mv = ith_minterm(*xv, mt=i)
                if mv and W[i, ell]: #compute constant terms
                    out[ell] = np.bitwise_or(out[ell], np.bitwise_and(mv, cs[W[i, ell]-1]))
        return out
    return fopt