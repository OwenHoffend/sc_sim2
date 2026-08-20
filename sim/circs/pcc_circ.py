import numpy as np
from sim.circs.circs import CombCirc
from sim.PCC import MMC

class Shared_RNS_PCC(CombCirc):
    def __init__(self, w, m, name="PCC_Circ"):
        self.w = w
        self.m = m
        super().__init__(m * w, m, w, name)

    def run(self, bs_mat):
        N = bs_mat.shape[1]
        rns = bs_mat[-self.w:, :]
        Xs = np.empty((self.m, N), dtype=np.bool_)
        for i in range(self.m):
            for j in range(N):
                Xs[i, j] = MMC(rns[:, j], bs_mat[self.w*i:self.w*(i+1), j], 1) #gamma = 1 corresponds to CMP
        return Xs
    
    def correct(self, parr):
        pass
    
    