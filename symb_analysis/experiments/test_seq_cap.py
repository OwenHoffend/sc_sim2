import sympy as sp
import numpy as np
from symb_analysis.seq_CAP import FSM_to_transition_matrix, transition_matrix_to_FSM, extend_markov_chain, get_steady_state
from sim.PTV import get_Q

def test_get_steady_state():
    x, y = sp.symbols('x y', real=True, nonneg=True)

    #Validated it on these three test cases from Armin's Exploiting Correlation paper:
    #T = sp.Matrix([[1-x, x, 0], [1-x, 0, x], [1-x, 0, x]])
    
    #T = sp.Matrix([
    #    [1-x, x, 0, 0],
    #    [1-x, 0, x, 0],
    #    [0, 1-x, 0, x],
    #    [0, 0, 1-x, x]
    #])

    #T = sp.Matrix([
    #    [1-x*y, x*y],
    #    [(1-x)*(1-y), x+y-x*y]
    #])

    #Transition matrix for a D-flipflop modeled with zero time steps of history
    #T = sp.Matrix([
    #    [1-x, x],
    #    [1-x, x]
    #])
    #print(get_steady_state(T))

    #Transition matrix for a D-flipflop modeled with one time step of history
    #dv = np.array(sp.symbols('v0 v1 v2 v3', real=True, nonneg=True))
    #T = sp.Matrix([
    #    [dv[0]/(dv[0]+dv[2]), dv[2]/(dv[0]+dv[2]), 0, 0],
    #    [0, 0, dv[1]/(dv[1]+dv[3]), dv[3]/(dv[1]+dv[3])],
    #    [dv[0]/(dv[0]+dv[2]), dv[2]/(dv[0]+dv[2]), 0, 0],
    #    [0, 0, dv[1]/(dv[1]+dv[3]), dv[3]/(dv[1]+dv[3])],
    #])
    #print(get_steady_state(T))

    #Transition matrix for a D=1 FSM synchronizer with one time step of history
    

    #print(get_steady_state(T))


def test_seq_dv():
    #Get the DVs for -1, 0, and 1 auto-SCC
    x, xx = sp.symbols('x xx', real=True, nonneg=True)
    Q = get_Q(2, lsb='right')
    Qinv = sp.Matrix(np.linalg.inv(Q))
    vn1 = sp.nsimplify(Qinv @ sp.Matrix([1, x, x, sp.Max(2*x-1, 0)]))
    v0 = sp.nsimplify(Qinv @ sp.Matrix([1, x, x, x**2]))
    v1 = sp.nsimplify(Qinv @ sp.Matrix([1, x, x, x]))
    print(sp.latex(vn1))
    print(sp.latex(v0))
    print(sp.latex(v1))