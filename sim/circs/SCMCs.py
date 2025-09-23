import numpy as np
from sim.circs.circs import Circ

#Seqential re-correlators as implemented in V. Lee, A Alaghi, L. Ceze, 2018. 
#Correlation Manipulating Circuits for Stochastic Computing
def fsm_syncronizer_d(x1_bs, x2_bs, d, extend_length=False):
    N1 = x1_bs.size
    N2 = x2_bs.size
    assert N1 == N2
    assert d <= N1
    final_len = N1
    if extend_length:
        final_len += d
    z1_bs = np.zeros(final_len, dtype=np.bool_)
    z2_bs = np.zeros(final_len, dtype=np.bool_)

    state = 0
    for i in range(final_len):
        if i < N1:
            x1 = x1_bs[i]
            x2 = x2_bs[i]
        else:
            x1 = False
            x2 = False
        z1 = x1
        z2 = x2
        if x1 != x2: 
            if x1: #unpaired x1
                if state > 0:
                    z1 = True
                    z2 = True
                else:
                    z1 = state == -d
                    z2 = False
                state -= 1
            else: #unpaired x2
                if state < 0:
                    z1 = True
                    z2 = True
                else:
                    z1 = False
                    z2 = state == d
                state += 1
            if np.abs(state) > d:
                state = d * np.sign(state)
        z1_bs[i] = z1
        z2_bs[i] = z2
    return z1_bs, z2_bs

class C_FSM_SYNC(Circ):
    def __init__(self, d, extend_length=False):
        self.d = d
        self.extend_length = extend_length
        super().__init__(2, 2, 0, [0, 1], "FSM_SYNC")

    def run(self, bs_mat):
        return fsm_syncronizer_d(bs_mat[0, :], bs_mat[1, :], self.d, self.extend_length)
    
    def correct(self, parr):
        return parr