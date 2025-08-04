import numpy as np
from sim.PTM import B_mat

def overlap_prob_mat(px, C):
    """Essentially the inverse of the SCC mat: build a matrix consisting of the overlap probability"""
    n = px.size
    O = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                O[i, j] = px[i]
            elif C[i, j] > 0:
                O[i, j] = C[i, j] * (min(px[i], px[j]) - px[i]*px[j]) + px[i]*px[j]
            else:
                O[i, j] = C[i, j] * (px[i]*px[j] - max(px[i] + px[j] - 1, 0)) + px[i]*px[j]
    return O

def scc_sat_inf(px, C):
    """Use an iterative algorithm to decide whether a set of input conditions is satisfiabile at N=inf
        and, if so, return the vin
    """
    n = px.size
    O = overlap_prob_mat(px, C)
    Bn = B_mat(n)
    Bn_s = np.sum(Bn, axis=1)
    v = np.zeros((2 ** n))
    tot = 0.0
    for i in reversed(np.argsort(Bn_s)[1:]):
        row = np.expand_dims(Bn[i, :], axis=1)
        mask = row @ row.T

        nzi = np.nonzero(O)
        if nzi[0].size == 0:
            break

        m = np.min(O[nzi])
        res = O - m * mask
        if np.all(res >= 0):
            O -= m * mask
            v[i] = m
            tot += m

    if tot <= 1:
        print("Satisfiable")
        v[0] = 1 - tot
        return v
    else:
        print("Not satisfiable")
        return None