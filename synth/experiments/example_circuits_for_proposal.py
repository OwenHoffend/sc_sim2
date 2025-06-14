import numpy as np
from sim.PTM import *
from sim.PTV import *
from sim.circs.circs import *
from sim.sim import *

class Example_Circ_ThreeGates(Circ):
    def __init__(self):
        super().__init__(3, 3, 0, [0, 0, 0], "ThreeGates")

    def run(self, bs_mat):
        z1 = np.bitwise_or(bs_mat[0, :], bs_mat[1, :])
        z2 = np.bitwise_and(bs_mat[1, :], bs_mat[2, :])
        z3 = np.bitwise_xor(bs_mat[0, :], bs_mat[2, :])
        return np.array([z1, z2, z3])
    
    def correct(self, parr):
        #return np.array([
        #    np.maximum(parr[0], parr[1]),
        #    parr[1] * 0.5,
        #    parr[0] + 0.5 - 2 * parr[0] * 0.5
        #])
        return np.array([
            np.maximum(parr[0], parr[1]),
            np.minimum(parr[1], parr[2]),
            np.abs(parr[0] - parr[2])
        ])

#The MUX circuit example used in the COMAX paper
class Example_Circ_COMAX(Circ):
    def __init__(self):
        super().__init__(5, 2, 3, [0, 1], "COMAX_Example")

    def run(self, bs_mat):
        z1 = mux(bs_mat[0, :], bs_mat[2, :], bs_mat[4, :])
        z2 = mux(bs_mat[2, :], bs_mat[1, :], bs_mat[3, :])
        return np.array([z1, z2])
    
    def correct(self, parr):
        return np.array([0.5 * parr[0] + 0.25, 0.5 * parr[1] + 0.25])

class Example_Circ_MAC(Circ):
    def __init__(self):
        super().__init__(9, 2, 1, [x for x in range(8)], "MAC_Example")

    def run(self, bs_mat):
        m1 = np.bitwise_and(bs_mat[0, :], bs_mat[1, :])
        m2 = np.bitwise_and(bs_mat[2, :], bs_mat[3, :])
        m3 = np.bitwise_and(bs_mat[4, :], bs_mat[5, :])
        m4 = np.bitwise_and(bs_mat[6, :], bs_mat[7, :])
        z1 = mux(m1, m2, bs_mat[8, :])
        z2 = mux(m3, m4, bs_mat[8, :])
        return np.array([z1, z2])

    def correct(self, parr):
        return np.array([0.5 * (parr[0] * parr[1] + parr[2] * parr[3]),
                0.5 * (parr[4] * parr[5] + parr[6] * parr[7])])
    
class TWO_ANDs(Circ):
    def __init__(self):
        super().__init__(4, 2, 0, [], "Two ands")

    def run(self, bs_mat):
        return np.array([
            np.bitwise_and(bs_mat[0, :], bs_mat[1, :]),
            np.bitwise_and(bs_mat[2, :], bs_mat[3, :]),
        ])
    
    def correct(self, parr):
        return np.array([
            parr[0] * parr[1],
            parr[2] * parr[3]
        ])

class TWO_MUXs(Circ):
    def __init__(self):
        super().__init__(5, 2, 1, [], "Two ands")

    def run(self, bs_mat):
        return np.array([
            mux(bs_mat[0, :], bs_mat[1, :], bs_mat[4, :]),
            mux(bs_mat[2, :], bs_mat[3, :], bs_mat[4, :])
        ])
    
    def correct(self, parr):
        return np.array([
            0.5 * (parr[0] + parr[1]),
            0.5 * (parr[2] + parr[3])
        ])

class AND_WITH_NOT_CONST(Circ):
    def __init__(self):
        super().__init__(2, 1, 1, [], "AND with NOT")

    def run(self, bs_mat):
        return np.array([
            np.bitwise_and(bs_mat[0, :], np.bitwise_not(bs_mat[1, :]))
        ])
    
    def correct(self, parr):
        return np.array([parr[0] * 0.5])
    
class AND_WITH_CONST(Circ):
    def __init__(self):
        super().__init__(2, 1, 1, [], "AND with CONST")

    def run(self, bs_mat):
        return np.array([
            np.bitwise_and(bs_mat[0, :], bs_mat[1, :])
        ])
    
    def correct(self, parr):
        return np.array([parr[0] * 0.5])