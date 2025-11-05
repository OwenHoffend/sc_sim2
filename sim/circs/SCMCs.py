import numpy as np
import sympy as sp
from sim.circs.circs import SeqCirc

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
    return np.array([z1_bs, z2_bs])

class C_FSM_SYNC(SeqCirc):
    def __init__(self, d, extend_length=False):
        self.d = d
        self.extend_length = extend_length
        super().__init__(2, 2, 0, None, "FSM_SYNC")

    def run(self, bs_mat):
        return fsm_syncronizer_d(bs_mat[0, :], bs_mat[1, :], self.d, self.extend_length)

    def get_T(self, dv):
        #Transition matrix for the FSM
        #dv is the distribution vector for the inputs
        #dv[0] = p(x=0, y=0)
        #dv[1] = p(x=0, y=1)
        #dv[2] = p(x=1, y=0)
        #dv[3] = p(x=1, y=1)

        q = 2 * self.d + 1
        T = sp.Matrix.zeros(q, q)
        for i in range(q):
            for j in range(q):
                if i == j:
                    T[i, j] = dv[3] + dv[0]
                elif i == j + 1: #xy'
                    T[i, j] = dv[2]
                elif i == j - 1: #x'y
                    T[i, j] = dv[1]
        T[0, 0] = dv[3] + dv[0] + dv[2]
        T[q-1, q-1] = dv[3] + dv[0] + dv[1]
        return T

    def get_PTM_steady_state(self, pi):
        #pi is the steady state probability vector
        q = 2 * self.d + 1
        s0 = self.d
        assert len(pi) == q
        M = sp.Matrix.zeros(4, 4)
        M[0, 0] = 1
        M[3, 3] = 1
        M[1, 0] = sum(pi[s0:q-1])
        M[2, 0] = sum(pi[1:s0+1])
        M[1, 1] = pi[q-1] #sd
        M[2, 2] = pi[0]
        M[1, 3] = sum(pi[0:s0])
        M[2, 3] = sum(pi[s0+1:q])
        return M
    
    def correct(self, parr):
        return parr