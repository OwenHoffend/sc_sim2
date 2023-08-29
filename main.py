from sim.RNS import *
from sim.SA import *
from experiments.interpolation.interpolation import *

test_interp_1d(lambda x: (np.cos(x) + 1) / 2, 10, 5)