import numpy as np
from sim.circs.circs import *
from symb_analysis.seq_CAP import get_DV_symbols

class C_MULLER_C(SeqCirc):
    """Circ from: Stochastic Circuits for Computing Weighted
    Ratio With Applications to Multiclass Bayesian
    Inference Machine"""

    def __init__(self):
        super().__init__(2, 1, 0, 2, None, "MULLER_C")

    def get_transition_list(self):
        vars = self.get_vars()
        [xbyb, xby, xyb, xy] = get_DV_symbols(vars, 0)
        transitions = [
            (0, 0, xbyb + xby + xyb),
            (0, 1, xy),
            (1, 0, xbyb),
            (1, 1, xby + xyb + xy)
        ]
        return transitions

    def get_mealy_TTs(self):
        TTs = np.zeros((2 ** self.n, self.m, self.ns))
        TTs[:, :, 0] = 0
        TTs[:, :, 1] = 1
        return TTs

    def run(self, bs_mat):
        _, N = bs_mat.shape
        Z_mat = np.zeros((N,), dtype=np.bool_)
        Z_prev = 0
        for i in range(N):
            X = bs_mat[0, i]
            Y = bs_mat[1, i]
            if X == Y:
                Z_prev = X
                Z_mat[i] = X
            else:
                Z_mat[i] = Z_prev

        return Z_mat