from sim.RNS import *
from sim.SA import *
from sim.COMAX import COMAX
from sim.circs import mux
from experiments.et_variance import *

et_var_test(10000, 4, hypergeometric=False)