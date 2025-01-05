import numpy as np
import matplotlib.pyplot as plt

from sim.sim import *
from sim.SNG import *
from sim.PCC import *
from sim.RNS import *
from sim.datasets import *
from sim.circs.circs import *
from experiments.early_termination.RET import *

def ET_on_images():
    err_thresh = 0.04
    max_w = 8
    circ = C_RCED()

    ds = dataset_imagenet(2, mode='list', idxs=[14, ], num_imgs=1)
    #ds.disp_img(0)

    N_PRET, PRET_err, PRET_w = analyze_PRET(max_w, circ, ds, err_thresh)
    sng = PRET_SNG_WN(PRET_w, circ)
    sim_run = sim_circ(sng, circ, ds)

    ds.disp_output_img(sim_run.out, 0)