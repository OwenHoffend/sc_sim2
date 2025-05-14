import numpy as np
from synth.sat import *
from sim.PTM import *
from sim.Util import bin_array
from itertools import chain, combinations
from functools import reduce

def get_a(n):
    Bn = B_mat(n) #This function includes caching of the Bmat to improve runtime
    for i in range(2 ** n):
        yield tuple(Bn[i, :])

def get_D(n):
    s = list(range(n))
    for r in range(len(s) + 1):
        for c in combinations(s, r):
            yield c

Q_dict = {}
def get_Q(n):
    if n in Q_dict.keys():
        return Q_dict[n]
    Q = np.zeros((2 ** n, 2 ** n), dtype=np.bool_)
    Bn = B_mat(n)
    for k, Dk in enumerate(get_D(n)):
        cols = Bn[:, Dk]
        reduced = np.bitwise_and.reduce(cols, axis=1)
        Q[k, :] = reduced
    Q_dict[n] = Q
    return Q

def idx_min(Px, S):
    return min(Px[S])

def get_PTV(C, Px):
    """Full linear solution for a PTV - may be computationally inefficient"""

    n, _ = C.shape
    assert len(Px) == n

    #first check satisfiability
    sat_result = sat(C)
    if sat_result is None:
        return None
    (S, L, R) = sat_result
    n_star = len(S)

    Q = get_Q(n)
    Q_inv = np.linalg.inv(Q)

    #Compute the P vector
    P = np.zeros((2 ** n))
    for k, Dk in enumerate(get_D(n)):
        if k == 0:
            P[0] = 1
            continue
        prod = 1.0
        for i in range(n_star):
            sdk = set(Dk)
            SiDk = S[i].intersection(sdk)
            if SiDk == set():
                continue
            LiDk = L[i].intersection(sdk)
            RiDk = R[i].intersection(sdk)
            if LiDk == set() or RiDk == set():
                prod *= min(Px[list(SiDk)])
            else:
                prod *= max(min(Px[list(LiDk)]) + min(Px[list(RiDk)]) - 1, 0)
        P[k] = prod

    return Q_inv @ P

def get_C_from_v(v, invalid_corr=1):
    n = int(np.log2(v.size))
    Bn = B_mat(n)
    P = Bn.T @ v
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p_uncorr = P[i] * P[j]
            cov = (Bn[:, i] * Bn[:, j]) @ v - p_uncorr
            if cov > 0:
                norm = np.minimum(P[i], P[j]) - p_uncorr
            else:
                norm = p_uncorr - np.maximum(P[i] + P[j] - 1, 0)
            if norm == 0:

                #This is the "invalid corr" decision that Tim uses:
                if cov > 0:
                    C[i, j] = -1
                else:
                    C[i, j] = 1
                    
                #C[i, j] = invalid_corr
            else:
                C[i, j] = cov / norm
    return C

def get_Px_from_v(v):
    n = int(np.log2(v.size))
    Bn = B_mat(n)
    return Bn.T @ v
    