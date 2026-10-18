from sim.sim import *
from sim.SNG import *
from sim.PCC import *
from sim.SCC import *
from sim.RNS import *
from sim.datasets import *
from sim.circs.circs import *
from CAP.PTV import *
from CAP.PTM import *
from COOPT.COOPT import *
from CAP.sat import *
from experiments.joint_area_example import *
from experiments.example_circuits_for_proposal import *
from experiments.subcircuit_ptm_example import *
from experiments.proposal_analysis_examples import *
from COOPT.unit_tests.run_synth_tests import *
from CAP.PTV import get_Q
from experiments.early_termination.old_earlytermination.early_termination_plots import *
from experiments.early_termination.ET_on_images import *
import matplotlib.pyplot as plt
from experiments.sympy_test import *
from sim.sim import sim_circ
from experiments.sequential.scmc_test import *
from sim.circs.SCMCs import C_FSM_SYNC
from CAP.seq_CAP import get_DV_symbols
from experiments.sequential.test_autocorr_bitstreams import *
from experiments.test_nonint_scc_gen import *
from CAP.subcirc_ptm import *
from CAP.copula_modeling import *
from experiments.copula_paper_examples import *
from COOPT.unit_tests.COOPT_tests import *
from COOPT.branch_and_bound_single_output import *
from experiments.sequential.muller_c_tests import *
from experiments.sequential.test_seq_cap import *
from COOPT.circuits_graph import *
from dissertation.dissertation_examples import *

if __name__ == "__main__":
    C1 = np.array([
        [1, -0.4, 0.3], 
        [-0.4, 1, -0.3], 
        [0.3, -0.3, 1]
    ])
    C2 = np.array([
        [1, 0, -1],
        [0, 1, 0],
        [-1, 0, 1]
    ])
    from scipy.linalg import block_diag
    C = block_diag(C1, C2)
    print(C)

    s, A = get_acoeffs_and_signs_from_C(C)
    print(s)

    np.set_printoptions(suppress=True, precision=3)
    print(A.T)

    #test_nonint_scc_gen_3vars()