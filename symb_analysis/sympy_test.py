import sympy as sp
from sim.PTM import get_PTM, reduce_func_mat
from sim.circs.circs import *
from synth.experiments.example_circuits_for_proposal import XOR_with_AND
from synth.sat import sat
from sim.PTV import get_Q, get_D
import numpy as np

#Symbolic PTV representing a correlation matrix containing only 0 and 1
def get_sym_ptv(C, lsb='right'):
    n, _ = C.shape

    #check satisfiability
    sat_result = sat(C)

    if sat_result is None:
        return None
    (S, L, R) = sat_result
    n_star = len(S)

    Q = get_Q(n, lsb=lsb)
    Q_inv = sp.Matrix(np.linalg.inv(Q))
    print(Q_inv)

    Px = np.array([sp.Symbol(f'x{i+1}') for i in range(n)], dtype=object)

    def get_sp_vars(s):
        varlist = list(Px[list(s)])
        return tuple(varlist)

    #Compute the P vector
    P = sp.Matrix.zeros(2 ** n, 1)
    for k, Dk in enumerate(get_D(n)):
        if k == 0:
            P[0] = sp.Integer(1)
            continue
        prod = sp.Integer(1)
        for i in range(n_star):
            sdk = set(Dk)
            SiDk = S[i].intersection(sdk)
            if SiDk == set():
                continue
            LiDk = L[i].intersection(sdk)
            RiDk = R[i].intersection(sdk)
            if LiDk == set() or RiDk == set():
                prod *= sp.Min(*get_sp_vars(SiDk))
            else:
                prod *= sp.Max(sp.Min(*get_sp_vars(LiDk)) + sp.Min(*get_sp_vars(RiDk)) - 1, 0)
        P[k] = prod

    return sp.simplify(Q_inv @ P)

def test_sym_ptv():
    #C = np.ones((3, 3))
    C = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])
    vin = get_sym_ptv(C)
    ptm = sp.Matrix(1 * get_PTM(XOR_with_AND(), lsb='right'))
    print(sp.simplify(sp.Matrix(ptm.T @ vin)))

def mux_pair_test():
    C = np.ones((4, 4))
    #C = np.eye(4)
    vin = get_sym_ptv(C)
    ptm = get_PTM(C_MUX_PAIR(), lsb='right')
    ptm_r = sp.Matrix(reduce_func_mat(ptm, 0, 0.5))
    print(sp.simplify(sp.Matrix(ptm_r.T @ vin)))

def maj_pair_test():
    C = np.ones((4, 4))
    #C = np.eye(4)
    vin = get_sym_ptv(C)
    ptm = get_PTM(C_MAJ_PAIR(), lsb='right')
    ptm_r = sp.Matrix(reduce_func_mat(ptm, 0, 0.5))
    print(sp.simplify(sp.Matrix(ptm_r.T @ vin)))

def test_ptv_ptm():
    x, y = sp.symbols('x y')
    vin = sp.Matrix([
        sp.Max(0, x + y - 1),
        sp.Max(0, x - y),
        sp.Max(0, y - x),
        sp.Min(x, y)
    ])

    ptm = sp.Matrix(1 * get_PTM(C_XOR(), lsb='right'))
    print(sp.simplify(sp.Matrix(ptm.T @ vin)))