import numpy as np
import sympy as sp
from sim.circs.circs import SeqCirc
from symb_analysis.seq_CAP import get_DV_symbols

class C_DFF_MEALY(SeqCirc):
    def __init__(self):
        super().__init__(1, 1, 0, 2, None, "DFF_MEALY")

    def get_transition_list(self):
        x, xb = sp.symbols("x xb")
        return [(0, 1, x), (0, 0, xb), (1, 0, xb), (1, 1, x)]

    def get_mealy_TTs(self):
        TTs = np.zeros((2 ** self.n, self.m, self.ns))
        TTs[:, 0, 0] = np.array([1, 0])
        TTs[:, 0, 1] = np.array([0, 1])
        return TTs

    def run(self, bs_mat):
        s = 0
        bs_out = np.zeros_like(bs_mat)
        for idx, b in enumerate(bs_mat[0, :]):
            if s == 0:
                bs_out[0, idx] = 1-b
            else:
                bs_out[0, idx] = b
            if b:
                s = 1
            else:
                s = 0
        return bs_out
