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
from synth.experiments.proposal_analysis_examples import *
from synth.unit_tests.run_synth_tests import *
from sim.PTV import get_Q
from experiments.early_termination.old_earlytermination.early_termination_plots import *
from experiments.early_termination.ET_on_images import *
import matplotlib.pyplot as plt
import numpy as np
from symb_analysis.sympy_test import *
from sim.sim import sim_circ
from experiments.sequential.scmc_test import sim_fsm_sync, test_CAP_fsm_sync, synchronizer_symbolic_curves, synchronizer_symbolic_error, sim_fsm_sync_px_sweep
from sim.circs.SCMCs import C_FSM_SYNC
from symb_analysis.experiments.test_seq_cap import *
from symb_analysis.seq_CAP import get_DV_symbols

if __name__ == "__main__":
    #sim_fsm_sync()
    sim_fsm_sync_px_sweep()
    #test_FSM_SYNC()
    #lfsr_autocorrelation_simulation_1d()
    #test_get_extended_mealy_ptm_DFF()