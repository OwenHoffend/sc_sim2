import sympy as sp
import numpy as np
from sim.PTM import B_mat, get_PTM
from synth.sat import sat
from sim.PTV import get_Q, get_D
from sim.Util import clog2
from sim.PTM import reduce_func_mat

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

    Px = np.array([sp.Symbol(f'x{i+1}', real=True, nonneg=True) for i in range(n)], dtype=object)

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

    return sp.nsimplify(Q_inv @ P)

def marginalize_ptv(ptv, vars_to_marginalize, lsb='right'):
    n = clog2(ptv.shape[0])
    Bn = B_mat(n, lsb=lsb)
    total = sp.Integer(0)
    for i in range(2 ** n):
        a = Bn[i, :]
        if all(a[vars_to_marginalize]):
            total += ptv[i]
    return sp.simplify(total)

def get_scc(pxixj, pxi, pxj, norm_name):

    #Special integer SCC cases
    if(pxixj == sp.Min(pxi, pxj)):
        return sp.Integer(1)
    elif(pxixj == sp.Max(pxi + pxj - 1, 0)):
        return sp.Integer(-1)
    elif(pxixj == pxi * pxj):
        return sp.Integer(0)

    #Otherwise, just create a symbol for the normalization term
    #TODO: Implement better simplification of the actual norm term instead of just making a symbol

    norm = sp.Symbol(f'{norm_name}')
    cov = sp.simplify(pxixj - pxi * pxj)
    return cov / norm

def CAP_analysis(circ, C, lsb='right', mode="full"):
    vin = get_sym_ptv(C, lsb=lsb)
    ptm = get_PTM(circ, lsb=lsb)

    if circ.nc > 0:
        for i in range(circ.nc):
            ptm = reduce_func_mat(ptm, i, 0.5)
    else:
        ptm = ptm * 1
    ptm = sp.Matrix(ptm)
    vout = sp.nsimplify(sp.Matrix(ptm.T @ vin)) #nsimplify here gets rid of the 1.0 terms in the expressions

    if mode == "full":
        Pz = sp.Matrix([marginalize_ptv(vout, [i], lsb=lsb) for i in range(circ.m)])
        if circ.m == 1:
            return Pz
        else:
            Cout = sp.Matrix.eye(circ.m, circ.m)
            for i in range(circ.m):
                for j in range(circ.m):
                    if j >= i:
                        break
                    else:
                        pzizj = marginalize_ptv(vout, [i, j], lsb=lsb)
                        c = get_scc(pzizj, Pz[i], Pz[j], norm_name=f'N({i},{j})')
                        Cout[i, j] = c
                        Cout[j, i] = c
            return Cout, Pz
    elif mode == "ptv":
        return vout
    else:
        print("Invalid mode")