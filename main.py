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
from experiments.sequential.test_autocorr_bitstreams import *
from experiments.test_nonint_scc_gen import *
from symb_analysis.experiments.subcirc_ptm import *
from sim.copula_modeling import *
from synth.experiments.copula_paper_examples import *
from synth.unit_tests.COOPT_tests import *
from synth.branch_and_bound_single_output import *
import unittest

if __name__ == "__main__":
    #result = branch_and_bound_opt_single_output(np.array([0, 1, 1, 2]))
    #result = branch_and_bound_opt_single_output(np.array([9, 5, 6, 0])) #Example of one that has two "new bests"
    #result = branch_and_bound_opt_single_output(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])) #Example of one that has two "new bests"
    joint_area_example_DV_based()

    #c1 = CubePairMulti(Cube(2, 0b11, 0b11), Cube(4, 0b1, 0b0), 3)
    #c2 = CubePairMulti(Cube(2, 0b0, 0b0), Cube(4, 0b1111, 0b1), 3)
    #c3 = CubePairMulti(Cube(2, 0b11, 0b1), Cube(4, 0b111, 0b0), 3)
    #c4 = CubePairMulti(Cube(2, 0b11, 0b10), Cube(4, 0b111, 0b0), 3)
    #c5 = CubePairMulti(Cube(2, 0b11, 0b1), Cube(4, 0b10, 0b10), 2)
    #c6 = CubePairMulti(Cube(2, 0b11, 0b11), Cube(4, 0b111, 0b11), 2)
    #cube_pairs = [c1, c2, c3, c4, c5, c6]
    #print(get_row_MVs_from_SEMs(convert_cube_pairs_to_SEMs(cube_pairs, 2, 4, 2)))

    #test_mask = 0b1100_0000
    #test_mask_slider = 1
    #arr = np.zeros(8, dtype=np.bool_)
    #for i in range(8):
    #    if test_mask & test_mask_slider:
    #        arr[i] = True
    #    test_mask_slider <<= 1
#
    #print(arr)