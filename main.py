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
from experiments.early_termination.et_gamma import *

if __name__ == "__main__":
    w = 6
    Nrange = [2 ** x for x in range(2, w)]
    #ds = dataset_discrete(1000, 1, np.array([0.0, 0.5]), np.array([0.5, 0.5]))
    ds = dataset_img_windows("./data/cameraman.png", 1, num=1000)
    ET_MSE_vc_N(C_WIRE(), ds, Nrange, w)