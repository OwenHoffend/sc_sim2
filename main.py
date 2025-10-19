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
from symb_analysis.seq_CAP import test_get_steady_state, get_steady_state, test_seq_dv
from experiments.sequential.scmc_test import test_fsm_sync, test_CAP_fsm_sync, test_symbolic_fsm
from sim.circs.SCMCs import C_FSM_SYNC

if __name__ == "__main__":
    #test_seq_dv()

    x_vals = np.linspace(0, 1, 200)
    y1 = np.maximum(0, 2 * x_vals - 1)
    y2 = x_vals ** 2
    y3 = x_vals
    plt.figure()
    plt.plot(x_vals, y1, label="Function at ASCC=-1: max(0, 2PX-1)")
    plt.plot(x_vals, y2, label="Function at ASCC=0: PX^2")
    plt.plot(x_vals, y3, label="Function at ASCC=1: PX")
    plt.xlabel("PX")
    plt.ylabel("PZ")
    plt.title("Plot of functions at ASCC=-1, 0, and 1")
    plt.legend()
    plt.grid(True)
    plt.show()

    #test_get_steady_state()
    #C = C_FSM_SYNC(2)
    #v0, v1, v2, v3 = sp.symbols('v0 v1 v2 v3', real=True, nonneg=True)
    #dv = np.array([v0, v1, v2, v3])
    #T = C.get_T(dv)
    #pi = get_steady_state(T, vars=[v0, v1, v2, v3])
    #print(pi)

    #sn1, s0, s1 = sp.symbols('sn1 s0 s1', real=True, nonneg=True)
    #pi = np.array([sn1, s0, s1])
    #M = C.get_PTM_steady_state(pi)
    #print(M)

    #test_CAP_fsm_sync()
    #test_symbolic_fsm()