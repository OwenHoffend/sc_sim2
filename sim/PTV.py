import numpy as np
from synth.sat import *
from sim.PTM import B_mat
from sim.Util import bit_vec_to_int
from sim.SCC import Pearson_to_SCC
from itertools import combinations
import sympy as sp

def tree_idx(n):
    #n=1: [[0,]]
    #n=2: [[0, 0], [0, 1]]
    #n=3: [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2]]
    if n == 1:
        return [[0, ]]
    else:
        seq = tree_idx(n-1)
        new_seq = []
        for subseq in seq:
            for i in range(n):
                new_seq.append(subseq + [i])
        return new_seq

def get_PTV_from_acoeffs_and_signs(acoeffs, signs, Px, lsb='right'):
    n = len(signs)
    P = np.zeros((2 ** n))
    for rns_choices in tree_idx(n):
        prob = 1.0
        for i in range(n):
            prob *= acoeffs[i][rns_choices[i]]
        
        if prob == 0:
            continue

        #Compute the L and R sets for the current RNS choices here
        print(rns_choices) #nonzero probability choices
        n_star = len(np.unique(rns_choices))
        L_ = [set() for _ in range(n)]
        R_ = [set() for _ in range(n)]
        for i in range(n):
            if signs[i] == 1: #left
                L_[rns_choices[i]].add(i)
            else: #right
                R_[rns_choices[i]].add(i)

        L = [set() for _ in range(n_star)]
        R = [set() for _ in range(n_star)]
        idx = 0
        for i in range(n):
            if L_[i] == set() and R_[i] == set():
                continue
            L[idx] = L_[i]
            R[idx] = R_[i]
            idx += 1

        for k, Dk in enumerate(get_D(n)):
            if k == 0:
                P[0] += 1 * prob
                continue
            prod = 1.0
            for i in range(n_star):
                LiDk = L[i].intersection(Dk)
                RiDk = R[i].intersection(Dk)
                prod *= max(min(list(Px[list(LiDk)]) + [1]) + min(list(Px[list(RiDk)]) + [1]) - 1, 0)
            P[k] += prod * prob

    Q = get_Q(n, lsb=lsb)
    Q_inv = np.linalg.inv(Q)
    return Q_inv @ P

def get_actual_PTV(bs_mat, delay=0, lsb='right'):
    n, N = bs_mat.shape

    if delay > 0:
        new_n = n * (delay + 1)
        new_bs_mat = np.zeros((new_n, N), dtype=np.bool_)
        for i in range(n):
            for j in range(delay + 1):
                new_bs_mat[i*(delay + 1) + j, :] = np.roll(bs_mat[i, :], j)

        n = new_n
        bs_mat = new_bs_mat

    Vin = np.zeros(2 ** n)
    uniques, counts = np.unique(bs_mat.T, axis=0, return_counts=True)
    for unq, cnt in zip(uniques, counts):
        Vin[bit_vec_to_int(unq, lsb=lsb)] = cnt / N
    return Vin

def get_a(n):
    Bn = B_mat(n) #This function includes caching of the Bmat to improve runtime
    for i in range(2 ** n):
        yield tuple(Bn[i, :])

def get_D(n):
    s = list(range(n))
    for r in range(len(s) + 1):
        for c in combinations(s, r):    
            yield c

def copula_transform_matrix(Mf, lsb='left'):
    n2, m2 = Mf.shape
    n = int(np.log2(n2))
    m = int(np.log2(m2))
    Qn = get_Q(n, lsb=lsb)
    Qm = get_Q(m, lsb=lsb)
    return Qm @ (Mf.T * 1) @ np.linalg.inv(Qn)

def copula_transform(Mf):
    n2, m2 = Mf.shape
    n = int(np.log2(n2))
    m = int(np.log2(m2))
    lexical_input = sp.symbols(["x{}".format("".join(str(i) for i in d)) if d != () else "1" for d in get_D(n)])
    lexical_output = sp.symbols(["z{}".format("".join(str(i) for i in d)) if d != () else "1" for d in get_D(m)])
    T = copula_transform_matrix(Mf)
    pout = sp.Matrix(T) @ sp.Matrix(lexical_input)
    pout_simp = []
    for idx, expr in enumerate(pout):
        expr = sp.nsimplify(expr)
        print(lexical_output[idx], "=", expr)
        pout_simp.append(expr)
    return pout_simp

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
    