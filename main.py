from sim.RNS import *
from sim.SA import *
from sim.PTM import *
from sim.COMAX import COMAX
from sim.circs import *
from sim.ReSC import *
from sim.ATPP import *
from experiments.et_hardware import *

#w = get_weight_matrix_from_ptm(get_func_mat(robert_cross, 5, 1), 1, 1, 4)[:, 0]
#print(w)

#gamma_correction()

r1 = RCED_et(8, 1000, "MNIST", staticN=512)
r2 = RCED_et(8, 1000, "MNIST", staticN=256)
r3 = RCED_et(8, 1000, "MNIST", staticN=128)
r4 = RCED_et(8, 1000, "MNIST", staticN=64)
r5 = RCED_et(8, 1000, "MNIST", staticN=32)
r6 = RCED_et(8, 1000, "MNIST", staticN=16)
print(r1)
print(r2)
print(r3)
print(r4)
print(r5)
print(r6)

#parr = np.array([0.25+0.125, 0.25+0.125]) #np.random.uniform(size=(4,)) --> 0.011
#w = 4
#Bx = parr_bin(parr, w, lsb='right')
#correct = fp_array(Bx)
#cgroups = np.array([1, 2])
#bs_mat = CAPE_ET(parr, w, cgroups, et=True)
#print(correct)
#print(np.mean(bs_mat, axis=1))
#print(scc_mat(bs_mat))
#print_precision_points(bs_mat)