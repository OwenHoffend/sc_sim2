from sim.RNS import *
from sim.SA import *
from experiments.interpolation.interpolation import *

test_interp_1d(lambda x: (np.cos(x) + 1) / 2, 'cubic', 10, 5)
#test_interp_1d(lambda x: np.clip(np.tan(x), 0, 1), 'cubic', 10, 5)