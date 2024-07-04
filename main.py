from sim.RNS import *
from sim.SA import *
from sim.PTM import *
from sim.COMAX import COMAX
from sim.circs import *
from sim.ReSC import *
from img.img_io import *
from sim.ATPP import *
from sim.datasets import *
from experiments.early_termination.precision_analysis import *
from experiments.early_termination.et_hardware import *
from experiments.early_termination.early_termination_plots import *
from experiments.early_termination.et_ed import *
from experiments.early_termination.SET import *

if __name__ == "__main__":
    
    #max_var = 0.001

    #w = 8
    #Nrange = [static_ET(circ, ds, w, max_var=max_var), ]

    #Nrange = [2 ** x for x in range(2, w)]
    #ds = dataset_discrete(1000, 1, np.array([0.0, 0.5]), np.array([0.5, 0.5]))
    #ds = dataset_img_windows("./data/cameraman.png", 1, num=1000)

    #ds = dataset_imagenet_samples(10, 100, 2)
    #SC_vals, var_et_vals, var_et_Ns, cape_et_vals, cape_et_Ns, LD_et_vals, LD_et_Ns \
    #    = ET_sim(C_RCED(), ds, 128, 8, max_var=max_var)
    
    #np.save("results/bsds500_3063_SC.npy", np.array(SC_vals).reshape(320, 480))

    #static_ET(C_Gamma(), ds, w)

    #conf_mat_bsds500_rced(8068, load=True)
    #conf_mat_bsds500_rced(29030, load=True)
    #conf_mat_bsds500_rced(223060, load=True)
    #conf_mat_bsds500_rced(235098, load=True)

    #correct = 0.33
    #thresh = 0.05
    #bs_mat = lfsr_sng_efficient(np.array([correct, ]), 256, 8, pack=False)
    #optimal_ET(bs_mat, correct, thresh)

    #test_SET_hypergeo(100, 0.01)
    #test_SET_hypergeo_2input(10, 0.01, 20, 32) #<--- bits of precision
    #SET_px_sweep(25, [0.1, 0.05, 0.01])
    #SET_hypergeometric_px_sweep(100, [0.1, 0.05, 0.01])

    #error_breakdown(1000)

    #ds = dataset_uniform(1000, 2)
    #circ = C_AND_N(2)

    #ds = dataset_single_image("./data/cameraman.png", 2)
    #circ = C_RCED()

    #ideal_SET(ds, circ, 0.02, 0.020001)