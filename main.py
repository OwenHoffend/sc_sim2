from sim.sim import *
from sim.SNG import *
from sim.PCC import *
from sim.SCC import *
from sim.RNS import *
from sim.datasets import *
from sim.circs.circs import *
from sim.PTV import *
from sim.PTM import *
from synth.COOPT import *
from synth.sat import *
from synth.experiments.joint_area_example import *
from synth.experiments.example_circuits_for_proposal import *
from synth.experiments.COMAX_example import *
from synth.experiments.subcircuit_ptm_example import *
from synth.unit_tests.run_synth_tests import *

if __name__ == "__main__":
    #run_all_tests()
    sub_circuit_PTM_example()