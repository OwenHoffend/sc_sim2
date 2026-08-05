import numpy as np
from sim.circs.circs import *
from symb_analysis.seq_CAP import get_DV_symbols
from sim.SNG import RAND_SNG
from sim.SCC import ascc_from_bs
from scipy.stats import entropy

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

    def correct(self, parr):
        return parr[0] * parr[1] / (parr[0] * parr[1] + (1 - parr[0]) * (1 - parr[1]))

class C_MULLER_C_cascade(SeqCirc):
    #Linearly chain of k Muller-C elements

    def __init__(self, k):
        self.k = k
        super().__init__(k + 1, 1, 0, 2 ** k, None, "MULLER_C_cascade")

    def get_transition_list(self):
        raise NotImplementedError("get_transition_list not implemented for C_MULLER_C_cascade")

    def get_mealy_TTs(self):
        raise NotImplementedError("get_mealy_TTs not implemented for C_MULLER_C_cascade")

    def run(self, 
        bs_mat, 
        regen_between_layers=False, 
        measure_autocorrelation_between_layers=False, 
        measure_err_between_layers=False, 
        w=0,
        correct_values=None
    ):
        Z_mat = bs_mat[0, :]
        _, N = bs_mat.shape
        asccs = np.zeros(self.k-1)
        errs = np.zeros(self.k-1)
        for i in range(1, self.k):
            if regen_between_layers:
                Zmean = np.mean(Z_mat)
                sng = RAND_SNG(w, np.eye(1))
                Z_mat = sng.run(Zmean, N)[0]
            Z_mat = C_MULLER_C().run(np.array([Z_mat, bs_mat[i, :]]))
            if measure_autocorrelation_between_layers:
                asccs[i-1] = ascc_from_bs(Z_mat)

            if measure_err_between_layers:
                Z = np.mean(Z_mat)
                Z_distr = np.array([1-Z, Z])
                correct_distr = np.array([1-correct_values[i-1], correct_values[i-1]])
                errs[i-1] = entropy(Z_distr, correct_distr)
                #errs[i-1] = np.sqrt((Z - correct_values[i-1]) ** 2)
        return Z_mat, asccs, errs

    def correct(self, parr):
        p = np.prod(np.array(parr))
        pn = np.prod(1 - np.array(parr))
        return p / (p + pn)