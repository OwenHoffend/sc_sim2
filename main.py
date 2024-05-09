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
from experiments.early_termination.et_on_images import *
from experiments.early_termination.early_termination_plots import *
from experiments.early_termination.et_gamma import *
from experiments.early_termination.et_AND import *

if __name__ == "__main__":
    Nrange = [2 ** x for x in range(2, 10)]
    ET_MSE_vc_N(C_RCED(), dataset_center_beta(100, 4), Nrange, 8)