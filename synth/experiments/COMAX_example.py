import numpy as np
from sim.circs.circs import *
from sim.sim import *
from synth.COOPT import *
from sim.datasets import *
from synth.experiments.example_circuits_for_proposal import *
from sim.visualization import *

def COMAX_example():
    c = Example_Circ_COMAX()
    ds = dataset_uniform(1000, c.nv)
    Cin = np.ones((c.nv, c.nv))
    Cout = np.ones((c.m, c.m))
    circ_opt = COOPT_via_PTVs(c, Cout)
    sim_result_opt = sim_circ_PTM(circ_opt, ds, Cin, validate=True)
    sim_result_original = sim_circ_PTM(c, ds, Cin, validate=True)

    #Plot the results
    plot_scc_histogram(
        [sim_result_opt.scc_array(0, 1), sim_result_original.scc_array(0, 1)], 
        labels=["COOPT", "Original"],
        title="COMAX Example Circuit, SCC of Z1 and Z2",
        xlabel="SCC",
        ylabel="Frequency",
        bins=30
    )

