import sympy as sp
import numpy as np

def test_ptv_2vars(Cin):
    X, Y, C, N = sp.symbols('X Y C N', real=True, nonneg=True)
    p = sp.Matrix([1, X, Y, C*N+X*Y])
    Q = sp.Matrix([
        [1, 1, 1, 1], 
        [0, 1, 0, 1], 
        [0, 0, 1, 1], 
        [0, 0, 0, 1]])

    print(Q.inv())

    v = sp.simplify(Q.inv() @ p)
    print(v)

    #substitute Cin into C of norm function
    if Cin == 1:
        Norm = sp.Min(X, Y) - X*Y
    elif Cin == 0 or Cin == -1: #doesn't matter in this case because norm gets multiplied by 0
        Norm = X*Y - sp.Max(X+Y-1,0)
    else:
        raise ValueError(f"Invalid Cin: {Cin}")

    v = v.subs(N, Norm)
    v = v.subs(C, Cin)
    v = sp.simplify(v)
    print(v)