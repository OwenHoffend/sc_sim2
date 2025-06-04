from sim.sim import *
from sim.SNG import *
from sim.PCC import *
from sim.SCC import *
from sim.RNS import *
from sim.datasets import *
from sim.circs.circs import *
from sim.PTV import *
from sim.PTM import *
from sim.COOPT import *
from synth.sat import *

import unittest
import synth.unit_tests.sat_tests as SatTest

from synth.experiments.joint_area_example import *
from synth.experiments.corr_analysis import *
import time

if __name__ == "__main__":
    #suite = unittest.TestLoader().loadTestsFromModule(SatTest)
    #unittest.TextTestRunner().run(suite)

    #c = Example_Circ_ThreeGates()
    #ds = dataset_single([4/8, 7/8, 3/8])
    #Cin = np.ones((3, 3))

    #c = Example_Circ_COMAX()
    #ds = dataset_uniform(1000, c.nv)
    C = np.array([
        [1, 0],
        [0, 1]
    ])
    
    start_time = time.time()
    circ = Example_Circ_COMAX()
    COOPT_via_PTVs(circ, C)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")