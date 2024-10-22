import numpy as np
from sim.circs.circs import Circ, mux

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