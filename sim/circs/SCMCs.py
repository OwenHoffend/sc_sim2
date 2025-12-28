import numpy as np
import sympy as sp
from sim.circs.circs import SeqCirc
from symb_analysis.seq_CAP import get_DV_symbols
from sim.PTM import TT_to_ptm

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
        ns = 2 * d + 1
        self.extend_length = extend_length
        super().__init__(2, 2, 0, ns, None, "FSM_SYNC")

    def run(self, bs_mat):
        return fsm_syncronizer_d(bs_mat[0, :], bs_mat[1, :], self.d, self.extend_length)

    def get_transition_list(self):
        vars = ["x", "y"]
        [xbyb, xby, xyb, xy] = get_DV_symbols(vars, 0)
        transitions = [
            (0, 0, xbyb+xy+xyb),
            (self.ns-1, self.ns-1, xbyb+xy+xby)
        ]
        for i in range(1, self.ns):
            transitions.append((i, i-1, xyb))
            transitions.append((i-1, i, xby))
        for i in range(1, self.ns-1):
            transitions.append((i, i, xbyb+xy))

        return transitions

    def get_mealy_TTs(self):
        left_end_state_TT = np.array([
            [0, 0],
            [1, 1],
            [1, 0],
            [1, 1],
        ])
        left_state_TT = np.array([
            [0, 0],
            [1, 1],
            [0, 0],
            [1, 1]
        ])
        mid_state_TT = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 1],
        ])
        right_state_TT = np.array([
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1]
        ])
        right_end_state_TT = np.array([
            [0, 0],
            [0, 1],
            [1, 1],
            [1, 1],
        ])
        TTs = np.zeros((2 ** self.n, self.m, self.ns))
        TTs[:, :, 0] = left_end_state_TT
        TTs[:, :, self.ns-1] = right_end_state_TT
        for i in range(1, self.ns-1):
            if i < self.d:
                TTs[:, :, i] = left_state_TT
            elif i == self.d:
                TTs[:, :, i] = mid_state_TT
            else:
                TTs[:, :, i] = right_state_TT
        return TTs

    #For Bernoulli DV model
    def get_T(self, dv):
        #Transition matrix for the FSM
        #dv is the distribution vector for the inputs
        #dv[0] = p(x=0, y=0)
        #dv[1] = p(x=0, y=1)
        #dv[2] = p(x=1, y=0)
        #dv[3] = p(x=1, y=1)

        T = sp.Matrix.zeros(self.ns, self.ns)
        for i in range(self.ns):
            for j in range(self.ns):
                if i == j:
                    T[i, j] = dv[3] + dv[0]
                elif i == j + 1: #xy'
                    T[i, j] = dv[2]
                elif i == j - 1: #x'y
                    T[i, j] = dv[1]
        T[0, 0] = dv[3] + dv[0] + dv[2]
        T[self.ns-1, self.ns-1] = dv[3] + dv[0] + dv[1]
        return T

    #For Bernoulli DV model
    #def get_PTM_steady_state(self, pi):
    #    #pi is the steady state probability vector
    #    s0 = self.d
    #    assert len(pi) == self.ns
    #    M = sp.Matrix.zeros(4, 4)
    #    M[0, 0] = 1
    #    M[3, 3] = 1
    #    M[1, 0] = sum(pi[s0:self.ns-1])
    #    M[2, 0] = sum(pi[1:s0+1])
    #    M[1, 1] = pi[self.ns-1] #sd
    #    M[2, 2] = pi[0]
    #    M[1, 3] = sum(pi[0:s0])
    #    M[2, 3] = sum(pi[s0+1:self.ns])
    #    return M
    
    def correct(self, parr):
        return parr

def fsm_reco_abdellatef(x1_bs, x2_bs, d_CTR, d_ReCTR, impr1=False):
    #impr1 is the magnitude comparison (algorithm 1) method from [Iwagaki, 2025]
    
    N1 = x1_bs.size
    N2 = x2_bs.size
    assert N1 == N2

    CTR, ReCTR = 0, 0
    z1_bs, z2_bs = np.zeros_like(x1_bs), np.zeros_like(x1_bs)
    for i in range(N1):

        #Min/max determination
        cmp = False
        CTR = CTR + x1_bs[i] - x2_bs[i]

        #CTR saturation (signed)
        if CTR > 2 ** (d_CTR - 1) - 1:
            CTR = 2 ** (d_CTR - 1) - 1
        elif CTR < -2 ** (d_CTR - 1):
            CTR = -2 ** (d_CTR - 1)

        cmp = CTR > 0 or impr1 and (CTR == 0 and x1_bs[i] <= x2_bs[i])
        max_bs, min_bs = 0, 0
        if cmp:
            max_bs = x1_bs[i]
            min_bs = x2_bs[i]
        else:
            max_bs = x2_bs[i]
            min_bs = x1_bs[i]

        #Relocate algorithm
        max_bs_r, min_bs_r = max_bs, min_bs
        if min_bs and not max_bs:
            max_bs_r, min_bs_r = 0, 0
            ReCTR += 1

            #ReCTR saturation (unsigned)
            if ReCTR > 2 ** d_ReCTR - 1:
                ReCTR = 2 ** d_ReCTR - 1
        elif max_bs and not min_bs and ReCTR > 0:
            max_bs_r, min_bs_r = 1, 1
            ReCTR -= 1

        #Undo the swap
        #(was the swap necessary in the first place?)
        if cmp:
            z1_bs[i] = max_bs_r
            z2_bs[i] = min_bs_r
        else:
            z1_bs[i] = min_bs_r
            z2_bs[i] = max_bs_r

    return np.array([z1_bs, z2_bs])