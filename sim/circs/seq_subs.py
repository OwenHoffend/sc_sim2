import numpy as np
from sim.circs.circs import Circ, mux
from sim.Util import bs_delay, bs_extend

class SUB_MUX(Circ):
    """Fig. 2 from [Zhang, ASPCAS '22]"""

    def __init__(self):
        super().__init__(3, 1, 1, [0, 1, 2], "SUB_MUX")

    def run(self, bs_mat):
        return mux(bs_mat[0, :], np.bitwise_not(bs_mat[1, :]), bs_mat[2, :])
    
    def correct(self, parr):
        return 0.5 * (parr[0] - parr[1])
    
class IEU(Circ):
    """Iterative enhancement unit from [Liu, JETC '17]"""
    def __init__(self):
        super().__init__(2, 2, 0, [0, 1], "IEU")

    def run(self, bs_mat):
        n1 = np.bitwise_or(bs_mat[0, :], bs_mat[1, :])
        m1 = np.bitwise_and(bs_mat[0, :], bs_mat[1, :])
        m1d = bs_delay(m1, 1)
        n1e = bs_extend(n1, 1)
        return np.stack((m1d, n1e))

    def correct(self, parr):
        return [
            parr[0] + parr[1] - parr[0] * parr[1],
            parr[0] * parr[1]
        ]

class SUB_NOR_ITER(Circ):
    """Fig. 3 from [Liu, JETC '17]"""

    def __init__(self, stages=0):
        self.stages = stages
        if stages > 0:
            self.ieu = IEU()
        super().__init__(2, 1, 0, [0, 1], "SUB_NOR_ITER")

    def run(self, bs_mat):
        
        for stage in range(self.stages):
            bs_mat = 

    def correct(self, parr):
        return parr[0] - parr[1]