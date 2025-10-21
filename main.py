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
from experiments.sequential.scmc_test import test_fsm_sync, test_CAP_fsm_sync, test_symbolic_fsm
from sim.circs.SCMCs import C_FSM_SYNC
from symb_analysis.seq_CAP import FSM_to_transition_matrix, transition_matrix_to_FSM, extend_markov_chain_t1, get_steady_state, get_DV_symbols, get_steady_state_linear_system, get_steady_state_nullspace

if __name__ == "__main__":
    #Test of extended Markov chain on D-flipflop
    x, xb = sp.symbols("x xb")
    transitions = [(0, 1, x), (0, 0, xb), (1, 0, xb), (1, 1, x)]
    transitions = extend_markov_chain_t1(transitions, [x, xb])
    T = FSM_to_transition_matrix(4, transitions, vars=[x, xb])
    print(get_steady_state(T, vars=[x, xb]))
    print(get_steady_state_linear_system(T))
    print(get_steady_state_nullspace(T))

    #T = FSM_to_transition_matrix(4, transitions)
    #print(get_steady_state(T))

    #Test of extended Markov chain on FSM synchronizer
    [xy, xyb, xby, xbyb] = get_DV_symbols(["x", "y"], 0)
    delayed_vars = get_DV_symbols(["x", "y"], 1)
    transitions = [
        (0, 0, xbyb+xy+xyb),
        (1, 1, xbyb+xy),
        (2, 2, xbyb+xy+xby),
        (1, 0, xyb),
        (2, 1, xyb),
        (0, 1, xby),
        (1, 2, xby),
    ]
    transitions = extend_markov_chain_t1(transitions, [xy, xyb, xby, xbyb])
    T = FSM_to_transition_matrix(7, transitions, vars=delayed_vars)
    #print(get_steady_state(T, vars=delayed_vars))
    #print(get_steady_state_linear_system(T))
    print(get_steady_state_nullspace(T))
    pass