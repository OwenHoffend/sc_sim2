from sim.circs.circs import *
import numpy as np
from sim.PTM import *
from sim.PTV import *
from sim.sim import *
from sim.datasets import *
from sim.visualization import *

#NOTE:
#ALSO CHECK subcircuit_ptm_example.py for more analysis of these circuits

class MAC_ReLU(Circ):
    def __init__(self):
        super().__init__(6, 1, 1, [x for x in range(6)], "MAC_ReLU")

    def run(self, bs_mat):
        m1 = np.bitwise_not(np.bitwise_xor(bs_mat[0, :], bs_mat[1, :]))
        m2 = np.bitwise_not(np.bitwise_xor(bs_mat[2, :], bs_mat[3, :]))
        z1 = mux(m1, m2, bs_mat[5, :])
        z2 = np.bitwise_or(z1, bs_mat[4, :])
        return z2

    def correct(self, parr):
        return 0.5 * max(0, (parr[0] * parr[1] + parr[2] * parr[3]))
    
class MAC_with_Const(Circ):
    def __init__(self):
        super().__init__(6, 2, 1, [x for x in range(6)], "MAC_with_Const")

    def run(self, bs_mat):
        m1 = np.bitwise_not(np.bitwise_xor(bs_mat[0, :], bs_mat[1, :]))
        m2 = np.bitwise_not(np.bitwise_xor(bs_mat[2, :], bs_mat[3, :]))
        z1 = mux(m1, m2, bs_mat[5, :])
        z2 = bs_mat[4, :]
        return np.array([z1, z2])

    def correct(self, parr):
        return np.array([0.5 * (parr[0] * parr[1] + parr[2] * parr[3]), 0.5])
    

def MAC_ReLU_example():
    c = MAC_ReLU()
    c_with_const = MAC_with_Const()
    input_data = np.random.uniform(size=(1000, 6))
    np.hstack((input_data, 0.5 * np.ones((1000, 1))))
    Cin = np.array([
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
    ])
    result = sim_circ_PTM(c, Dataset(input_data), Cin, validate=True)
    result_with_const = sim_circ_PTM(c_with_const, Dataset(input_data), Cin, validate=True)
    scc = np.mean(result.scc_array(0, 1))
    scc_with_const = np.mean(result_with_const.scc_array(0, 1))
    print("MAC_ReLU RMSE: ", result.RMSE())
    print("MAC_ReLU scc: ", scc)
    print("MAC_with_Const RMSE: ", result_with_const.RMSE())
    print("MAC_with_Const scc: ", scc_with_const)
    