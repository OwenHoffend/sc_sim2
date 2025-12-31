import numpy as np
from synth.sat import *
from sim.PTM import B_mat
from sim.Util import bin_array, bit_vec_to_int
from itertools import chain, combinations
from functools import reduce

def get_actual_PTV(bs_mat):
    n, N = bs_mat.shape
    Vin = np.zeros(2 ** n)
    uniques, counts = np.unique(bs_mat.T, axis=0, return_counts=True)
    for unq, cnt in zip(uniques, counts):
        Vin[bit_vec_to_int(unq)] = cnt / N
    return Vin

def get_actual_DV_1cycle(bs):
    #given a bs, return the DV corresponding to it and its 1-cycle delayed counterpart
    xb_xb = np.mean(np.bitwise_and(
                np.bitwise_not(bs[0, :]), np.bitwise_not(np.roll(bs[0, :], 1))))
    xb_x = np.mean(np.bitwise_and(
                np.bitwise_not(bs[0, :]), np.roll(bs[0, :], 1)))
    x_xb = np.mean(np.bitwise_and(
                bs[0, :], np.bitwise_not(np.roll(bs[0, :], 1))))
    x_x = np.mean(np.bitwise_and(
                bs[0, :], np.roll(bs[0, :], 1)))
    return xb_xb, xb_x, x_xb, x_x

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
def get_Q(n, lsb='left'):
    if n in Q_dict.keys():
        return Q_dict[n]
    Q = np.zeros((2 ** n, 2 ** n), dtype=np.bool_)
    Bn = B_mat(n, lsb=lsb)
    for k, Dk in enumerate(get_D(n)):
        cols = Bn[:, Dk]
        reduced = np.bitwise_and.reduce(cols, axis=1)
        Q[k, :] = reduced
    Q_dict[n] = Q
    return Q

def idx_min(Px, S):
    return min(Px[S])

def get_PTV(C, Px, lsb='right'):
    """Full linear solution for a PTV - may be computationally inefficient"""

    n, _ = C.shape
    assert len(Px) == n

    #first check satisfiability
    sat_result = sat(C)

    #Some extra code to check axiom satisfiability
    sat_result_axiom = sat_via_axioms(C)
    if (sat_result is not None) != sat_result_axiom:
        print(sat_result)
        print(sat_result_axiom)
        raise ValueError("Sat result is incorrect")

    if sat_result is None:
        return None
    (S, L, R) = sat_result
    n_star = len(S)

    Q = get_Q(n, lsb=lsb)
    Q_inv = np.linalg.inv(Q)

    #Compute the P vector
    P = np.zeros((2 ** n))
    for k, Dk in enumerate(get_D(n)):
        if k == 0:
            P[0] = 1
            continue
        prod = 1.0
        for i in range(n_star):
            LiDk = L[i].intersection(Dk)
            RiDk = R[i].intersection(Dk)
            prod *= max(min(list(Px[list(LiDk)]) + [1]) + min(list(Px[list(RiDk)]) + [1]) - 1, 0)
        P[k] = prod

    return Q_inv @ P

#Special-case PTV generation
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

def get_vin_nonint_pair(c, px, py):
    vin_uncorr = get_vin_mc0(np.array([px, py]))
    if c >= 0:
        vin_corr = get_vin_mc1(np.array([px, py]))
        return c * vin_corr + (1 - c) * vin_uncorr
    else:
        vin_corr = get_vin_mcn1(np.array([px, py]))
        return -c * vin_corr + (1 + c) * vin_uncorr

def get_C_from_v(v, invalid_corr=1, return_P = False, lsb='right'):
    n = int(np.log2(v.size))
    Bn = B_mat(n, lsb=lsb) * 1.0
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
    if return_P:
        return P, C
    return C

def get_Px_from_v(v):
    n = int(np.log2(v.size))
    Bn = B_mat(n, lsb='right')
    return Bn.T @ v
    