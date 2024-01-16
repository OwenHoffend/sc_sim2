from sim.RNS import *
from sim.SA import *
from sim.PTM import *
from sim.COMAX import COMAX
from sim.circs import *
from experiments.et_variance import *

#w = np.array([0, 0.5, 0.5, 1]) #MUX/MAJ
#w = get_weight_matrix_from_ptm(get_func_mat(robert_cross, 5, 1), 1, 1, 4)[:, 0]
#print(w)
#var = 0.0001

#num_pxs = 100
#num_trials = 1

#RCED
#px11 = 0.1
#px12 = 0.1
#px21 = 0.9
#px22 = 0.9
#px = np.array([px11, px12, px21, px22])
#vin = get_vin_mc1(px)

#MUX
#px = 0.5
#py = 0.5
#px = np.array([px, py])
#vin = get_vin_mc0(px)

#sccs = [0, 1]
#vars = [0.001, 0.0001]

#scc_dynamic_et_test(w, var, num_pxs, num_trials, 1, 'MNIST_beta')
#binomial_dynamic_et_test(w, vin, var, num_trials, plot=True)

#bitstream_var_test_hypergeometric(10000, 16, 1024) #WORKS

#test of hypergeometric variance eq for full circuit
nv2 = 32
nc2 = 32

v_denom = 16
vin = np.random.random_integers(0, v_denom, size=(nv2,))
m = np.sum(vin)
vin = vin / m
w = np.random.random_integers(0, nc2, size=(nv2,)) / nc2
N = nc2 * m >> 5
get_MSE_hypergeometric(w, vin, N, nv2, nc2, m, 1000)