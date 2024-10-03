from sim.RNS import *
from sim.SA import *
from sim.PTM import *
from sim.COMAX import COMAX
from sim.circs import *
from sim.ReSC import *
from img.img_io import *
from sim.ATPP import *
from multiprocessing import Pool
from sim.datasets import *
from experiments.early_termination.precision_analysis import *
from experiments.early_termination.et_hardware import *
from experiments.early_termination.early_termination_plots import *
from experiments.early_termination.et_ed import *
from experiments.early_termination.SET import *
from analysis.scc_sat import *

if __name__ == "__main__":
    px = np.array([4/6, 4/6, 4/6])
    C = np.array([
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ])
    v = scc_sat_inf(px, C)
    print(v)