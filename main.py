from sim.RNS import *
from sim.SA import *
from sim.COMAX import COMAX
from sim.circs import mux
from experiments.et_variance import *

w = np.array([0, 0.5, 0.5, 1]) #MUX/MAJ
var = 0.0001

num_pxs = 1000
num_trials = 1
vin = np.array([1, 0, 0, 0])

#sccs = [0, 1]
#vars = [0.001, 0.0001]

#scc_dynamic_et_test(w, var, num_pxs, num_trials, 0, 'uniform')
binomial_dynamic_et_test(w, vin, var, num_trials, plot=True)