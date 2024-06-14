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

if __name__ == "__main__":
    
    #max_var = 0.001

    #print(np.mean(ds))
    #print(np.var(ds))

    #w = 8
    #Nrange = [static_ET(circ, ds, w, max_var=max_var), ]

    #Nrange = [2 ** x for x in range(2, w)]
    #ds = dataset_discrete(1000, 1, np.array([0.0, 0.5]), np.array([0.5, 0.5]))
    #ds = dataset_img_windows("./data/cameraman.png", 1, num=1000)

    #ds = dataset_imagenet_samples(1, 1000, 2)
    #SC_vals, var_et_vals, var_et_Ns, cape_et_vals, cape_et_Ns, LD_et_vals, LD_et_Ns \
    #= ET_sim(circ, ds, 128, 8, max_var=max_var)
    #np.save("results/bsds500_3063_SC.npy", np.array(SC_vals).reshape(320, 480))

    #static_ET(C_Gamma(), ds, w)

    conf_mat_bsds500_rced(8068, load=True)
    conf_mat_bsds500_rced(29030, load=True)
    conf_mat_bsds500_rced(223060, load=True)
    conf_mat_bsds500_rced(235098, load=True)