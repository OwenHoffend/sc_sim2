import numpy as np
from scipy.special import comb
from sim.Util import *
from sim.RNS import *
from sim.PCC import *
from sim.SCC import *

#Bernstein approximation from "An Architecture for Fault-Tolerant Computation with Stochastic Logic"
B_GAMMA = [
    0.0955,
    0.7207,
    0.3476,
    0.9988,
    0.7017,
    0.9695,
    0.9939
]

def ReSC(inputs):
    w, N = inputs.shape 
    sz = int((w-1) / 2)
    x = inputs[:sz, :]
    b = inputs[sz:, :]
    sel = np.sum(x, axis=0)
    z = np.take_along_axis(b, sel.reshape(1, N), axis=0)
    return z

def bernstein(coeffs, x):
    def B_i_n(i, n, x_):
        return comb(n, i) * (x_**i) * (1-x_)**(n-i)
    out = np.zeros_like(x)
    n = coeffs.size
    for i in range(n):
        out += coeffs[i] * B_i_n(i, n-1, x)
    return out

def ReSC_test():
    inputs = np.array([
        [True, False],
        [True, False],
        [True, True],

        [False, False],
        [True, True],
        [False, True],
        [True, False]
    ])
    ReSC(inputs)