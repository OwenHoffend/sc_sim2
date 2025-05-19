from sim.sim import *
from sim.SNG import *
from sim.PCC import *
from sim.SCC import *
from sim.RNS import *
from sim.datasets import *
from sim.circs.circs import *
from sim.PTV import *
from synth.sat import *

import unittest
import synth.unit_tests.sat_tests as SatTest

from synth.experiments.joint_area_example import *

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(SatTest)
    unittest.TextTestRunner().run(suite)

    #joint_area_example()