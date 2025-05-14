from sim.sim import *
from sim.SNG import *
from sim.PCC import *
from sim.SCC import *
from sim.RNS import *
from sim.datasets import *
from sim.circs.circs import *
from sim.PTV import *
from synth.sat import *

from experiments.early_termination.SET_error_model import *
from experiments.early_termination.ET_on_images import *

import unittest
import synth.tests.sat_tests as SatTest

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(SatTest)
    unittest.TextTestRunner().run(suite)