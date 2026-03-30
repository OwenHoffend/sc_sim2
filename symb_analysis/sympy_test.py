import sympy as sp
from sim.PTM import reduce_func_mat
from sim.circs.circs import *
from synth.experiments.example_circuits_for_proposal import *
import numpy as np
from symb_analysis.CAP import get_sym_ptv, marginalize_ptv, CAP_analysis

def test_sym_ptv():
    C = np.ones((3, 3))
    #C = np.array([
    #    [1, 1, 0],
    #    [1, 1, 0],
    #    [0, 0, 1]
    #])
    vin = get_sym_ptv(C)
    ptm = sp.Matrix(1 * XOR_with_AND_first_layer().get_PTM(lsb='right'))
    print(sp.simplify(sp.Matrix(ptm.T @ vin)))

def two_and_test():
    C = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    circ = TWO_ANDs()
    print(CAP_analysis(circ, C, lsb='right', mode="ptv"))

    #Run same test but using copula transform matrix instead
    Mf = circ.get_PTM()
    print(copula_transform(Mf))

def xor_and_first_layer_test():
    C = np.ones((3, 3))
    print(CAP_analysis(XOR_with_AND_first_layer(), C, lsb='right', mode="full"))

def mux_pair_test():
    C = np.ones((4, 4))
    circ = C_MUX_PAIR()
    vout = CAP_analysis(circ, C, lsb='right', mode="ptv")
    #print(sp.latex(2 * vout[3]))
    print(vout)

    Mf = circ.get_PTM()
    print(copula_transform(Mf))

def maj_pair_test():
    C = np.ones((4, 4))
    circ = C_MAJ_PAIR()
    vout = CAP_analysis(circ, C, lsb='right', mode="ptv")
    #print(sp.latex(2 * vout[3]))
    Mf = circ.get_PTM()
    print(copula_transform(Mf))

def test_ptv_ptm():
    x, y = sp.symbols('x y')
    vin = sp.Matrix([
        sp.Max(0, x + y - 1),
        sp.Max(0, x - y),
        sp.Max(0, y - x),
        sp.Min(x, y)
    ])

    ptm = sp.Matrix(1 * C_XOR().get_PTM(lsb='right'))
    print(sp.simplify(sp.Matrix(ptm.T @ vin)))

def piecewise_test():
    x = sp.Symbol('x', real=True, nonneg=True)
    y = sp.Symbol('y', real=True, nonneg=True)
    z = sp.Symbol('z', real=True, nonneg=True)

    z = sp.Min(x, y)
    print(z == sp.Min(y, x))

    p = sp.Piecewise(
        ((z - x*y) / (sp.Min(x, y) - x*y), z - x*y > 0),
        ((z - x*y) / (x*y - sp.Max(x+y-1, 0)), z - x*y <= 0)
    )
    print(p.simplify())
