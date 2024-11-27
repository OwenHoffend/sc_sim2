from sim.RNS import *
from sim.SA import *
from sim.PTM import *
from sim.COMAX import COMAX
from sim.circs.circs import *
from sim.ReSC import *
from img.img_io import *
from sim.ATPP import *
from sim.circs.seq_adders import *
from multiprocessing import Pool
from sim.datasets import *
from experiments.early_termination.precision_analysis import *
from experiments.early_termination.et_hardware import *
from experiments.early_termination.early_termination_plots import *
from experiments.early_termination.et_ed import *
from experiments.early_termination.SET import *
from experiments.entropy_autocorr import *
from analysis.scc_sat import *

if __name__ == "__main__":
    #bs_mat = lfsr_sng_precise_sample(np.array([0.33, 0.33]), 5, pack=False)
    fig_X()
    #SET_hypergeometric_px_sweep()
    #SET_hypergeometric_px_sweep(1000, [0.1, 0.05, 0.01])
    #test_basic_hypergeo(200)
    #test_basic_binomial(100)
