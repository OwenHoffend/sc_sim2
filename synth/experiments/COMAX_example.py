import numpy as np
from sim.circs.circs import *
from sim.sim import *
from synth.COOPT import *
from sim.datasets import *
from synth.experiments.example_circuits_for_proposal import *
from sim.visualization import *

def COMAX_example(): #TODO convert to unit test
    c = Example_Circ_COMAX()
    ds = dataset_uniform(1000, c.nv)
    #Cin = np.ones((c.nv, c.nv))
    Cin = np.identity(c.nv)
    #Cin = 2 * np.identity(c.nv) - np.ones((c.nv, c.nv))
    #Cout = np.ones((c.m, c.m))
    Cout = np.identity(c.m)
    #Cout = 2 * np.identity(c.m) - np.ones((c.m, c.m))
    circ_opt = COOPT_via_PTVs(c, Cout)
    sim_result_opt = sim_circ_PTM(circ_opt, ds, Cin, validate=True)
    sim_result_original = sim_circ_PTM(c, ds, Cin, validate=True)
    original_scc_array = sim_result_original.scc_array(0, 1)
    opt_scc_array = sim_result_opt.scc_array(0, 1)

    print("Average SCC of original circuit: ", np.mean(original_scc_array))
    print("Average SCC of optimized circuit: ", np.mean(opt_scc_array))

    #Plot the results
    plot_scc_histogram(
        [original_scc_array, opt_scc_array], 
        labels=["Original", "Optimized"],
        title="COOPT Example Circuit, SCC of Z1 and Z2",
        xlabel="SCC",
        ylabel="Frequency",
        bins=30
    )

