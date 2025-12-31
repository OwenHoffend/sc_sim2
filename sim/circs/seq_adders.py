import numpy as np
from sim.circs.circs import *
from symb_analysis.seq_CAP import get_DV_symbols

class ADD_TFF(Circ):
    """Fig. 9 from Stipcevic 2023"""
    def __init__(self):
        super().__init__(2, 1, 0, [0, 1], "ADD_TFF")

    def run(self, bs_mat):
        _, N = bs_mat.shape
        tff_seq = np.zeros((int(N/2), 2), dtype=np.bool_)
        tff_seq[:, 0] = True
        tff_seq = tff_seq.flatten() #alternating 0 and 1
        return mux(bs_mat[0, :], bs_mat[1, :], tff_seq)

    def correct(self, parr):
        return 0.5 * (parr[0] + parr[1])

class C_FSM_ADD(SeqCirc):
    """Circ from [Temenos, 2021]"""

    def __init__(self, M):
        if M < 2:
            raise ValueError("Minimum M is 3")
        self.M = M
        ns = M + 1
        super().__init__(2, 1, 0, ns, None, "FSM_ADD")

    def run(self, bs_mat):
        _, N = bs_mat.shape
        Tn = 0
        Z_mat = np.zeros((N,), dtype=np.bool_)
        for i in range(N):
            X = bs_mat[0, i]
            Y = bs_mat[1, i]
            Z_mat[i] = Tn > 0 or X or Y
            if X and Y and Tn < self.M - 1:
                Tn += 1
            elif not X and not Y and Tn > 0:
                Tn -= 1
        return Z_mat

    def get_transition_list(self):
        vars = self.get_vars()
        [xbyb, xby, xyb, xy] = get_DV_symbols(vars, 0)
        X = xyb + xy
        Y = xby + xy
        A = xbyb
        B = X + Y - 2 * xy
        C = xy
        transitions = [
            (0, 0, A), #0A state
            (0, 1, B),
            (1, 0, A),
            (0, 2, C),
            (self.M-1, self.M-1, B+C)
        ]
        for i in range(self.M - 1):
            transitions.append(i+1, i+2, C)
            transitions.append(i+2, i+1, A)
        return transitions

    def get_mealy_TTs(self):
        zero_A = np.array([
            [0],
            [0]
        ])
        all_others = np.array([
            [1],
            [1]
        ])
        TTs = np.zeros((2 ** self.n, self.m, self.ns))
        TTs[:, :, 0] = zero_A
        for i in range(1, self.ns):
            TTs[:, :, i] = all_others
        return zero_A