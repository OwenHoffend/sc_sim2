from sim.circs.circs import SeqCirc
import sympy as sp
import numpy as np

#For the stanh, the total number of states is twice the depth
def fsm_tanh(x_bs, d):
    N = x_bs.size
    z1_bs = np.zeros(N, dtype=np.bool_)
    state = 0
    for i in range(N):

        #Next state logic
        if x_bs[i]:
            if state < 2*d - 1:
                state += 1
        else:
            if state > 0:
                state -= 1

        #Output logic
        if state < d:
            z1_bs[i] = False
        else:
            z1_bs[i] = True
    return z1_bs

class C_TANH(SeqCirc):
    def __init__(self, d):
        self.d = d
        super().__init__(1, 1, 0, None, "TANH")

    def run(self, bs_mat):
        return fsm_tanh(bs_mat[0, :], self.d)

    def correct(self, parr):
        return 0.5 * (1 + sp.tanh(2*(2*parr[0]-1)))

    def get_T(self, dv):
        pass

    def get_PTM_steady_state(self, pi):
        pass