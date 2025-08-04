import numpy as np
import matplotlib.pyplot as plt

from sim.sim import *
from sim.SNG import *
from sim.PCC import *
from sim.RNS import *
from sim.datasets import *
from sim.circs.circs import *
from experiments.early_termination.RET import *
from experiments.early_termination.SET_error_model import *
from multiprocess import Pool

def ET_on_cameraman(err_thresh, max_w):
    circ = C_RCED()
    ds = dataset_cameraman(2)

    PRET_err, PRET_w = get_PRET_w(max_w, circ, ds, err_thresh)
    print("PRET_w: ", PRET_w)

    #PRET 
    sng_pret = PRET_SNG(PRET_w, circ, lzd=True)
    sim_run_pret = sim_circ(sng_pret, circ, ds, loop_print=True)
    ds.disp_output_img(1.0 - sim_run_pret.out, 0)

    Nset_hyper = sim_run_pret.avg_N()

    #LFSR + Hypergeo
    sng_lfsr = LFSR_SNG(max_w, circ)
    sim_run_lfsr = sim_circ(sng_lfsr, circ, ds, Nset=np.round(Nset_hyper).astype(np.int32), loop_print=True)
    ds.disp_output_img(1.0 - sim_run_lfsr.out, 0)

    print("LFSR + Hyper SET RMSE: ", sim_run_lfsr.RMSE())
    print("LFSR + Hyper SET avg N: ", sim_run_lfsr.avg_N())
    print("PRET RMSE", sim_run_pret.RMSE())
    print("PRET avg N", sim_run_pret.avg_N())

    pass

#Analysis for RET paper applications section
def ET_on_imagenet(circ, ds, idx, err_thresh, max_w, Nset_hyper=None, PRET_w=None):
    print("Image idx: ", idx)

    #ds.disp_img(0)

    #Van der Corput with Hypergeometric SET
    if Nset_hyper is None:
        Nset_hyper = SET_hyper(max_w, circ, ds, err_thresh)
    print("Nset_hyper: ", Nset_hyper)

    if PRET_w is None:
        N_PRET, PRET_err, PRET_w = get_PRET_w(max_w, circ, ds, err_thresh)
    print("PRET_w: ", PRET_w)

    #LFSR + Hypergeo
    sng_lfsr = LFSR_SNG(max_w, circ)
    sim_run_lfsr = sim_circ(sng_lfsr, circ, ds, Nset=np.round(Nset_hyper).astype(np.int32), loop_print=False)

    #BPC SET
    sng_bpc = PRET_SNG(PRET_w, circ, et=False)
    sim_run_bpc = sim_circ(sng_bpc, circ, ds, loop_print=False)
    #ds.disp_output_img(1.0 - sim_run_bpc.out, 0)

    #PRET
    sng_pret = PRET_SNG(PRET_w, circ, lzd=True)
    sim_run_pret = sim_circ(sng_pret, circ, ds, loop_print=False)
    #ds.disp_output_img(1.0 - sim_run_pret.out, 0)

    print("IDX: {} ".format(idx) + "LFSR + Hyper SET RMSE: ", sim_run_lfsr.RMSE())
    print("IDX: {} ".format(idx) + "LFSR + Hyper SET avg N: ", sim_run_lfsr.avg_N())
    print("IDX: {} ".format(idx) + "BPC SET RMSE ", sim_run_bpc.RMSE())
    print("IDX: {} ".format(idx) + "BPC SET avg N", sim_run_bpc.avg_N())
    print("IDX: {} ".format(idx) + "PRET RMSE", sim_run_pret.RMSE())
    print("IDX: {} ".format(idx) + "PRET avg N", sim_run_pret.avg_N())

    #ds.disp_output_img(sim_run_pret.Ns, 0, scale=False, colorbar=True)

    return sim_run_lfsr.RMSE(), sim_run_lfsr.avg_N(), sim_run_bpc.RMSE(), sim_run_bpc.avg_N(), sim_run_pret.RMSE(), sim_run_pret.avg_N()

def ET_on_imagenet_mp():
    NUM_CORES = 10
    img_list = list(range(2))

    #SET on entire dataset
    ds = dataset_mnist_beta(1000, 1)
    ds = ds.merge(dataset_all_same(ds.num, 1, 0.5))
    circ = C_MAX()
    err_thresh = 0.02
    max_w = 8

    print("Calculating PRET w")
    PRET_err, PRET_w = get_PRET_w(max_w, circ, ds, err_thresh, use_cache=True)

    print("Calculating SET")
    Nset_hyper = SET_hyper(PRET_w, circ, ds, err_thresh, use_cache=True, use_pow2=True)

    def f(i):
        return ET_on_imagenet(circ, ds, i, err_thresh, max_w, Nset_hyper=Nset_hyper, PRET_w=PRET_w)

    with Pool(NUM_CORES) as p:
        results = p.map(f, img_list)

    filename = "first_2.npy"
    arr = np.array(results)
    np.save("./results/relu_imagenet/set_entire_dataset/{}".format(filename), arr)

    #First run:
    #0 = (0.009998895606612577, 85.0, 0.010032511037475267, 64.0, 0.010032511037475267, 21.760022719235263)
    #1 = (0.01563836515145532, 33.0, 0.015117753900524378, 32.0, 0.015117753900524378, 5.608651037013787)

def plot_ET_on_imagenet_mp_results():
    #plotting/analysis function for ET on imagenet images
    
    #First load all of the daata

    to_view_1 = np.load("./results/rced_imagenet/set_entire_dataset/first_500.npy")
    to_view_2 = np.load("./results/rced_imagenet/set_entire_dataset/first_500_2.npy")

    plt.scatter(to_view_2[:, 1], to_view_2[:, 0], s=5, label="LFSR SET")
    #plt.scatter(to_view_2[:, 3], to_view_2[:, 2], s=5, label="BPC SET")
    plt.scatter(to_view_1[:, 5], to_view_1[:, 4], c="limegreen", s=5, label="PRET")
    plt.plot(np.mean(to_view_2[:, 1]), np.mean(to_view_2[:, 0]), '^', color='r', label=r"LFSR SET Average. $N_{SET}=$" + "{}".format(np.mean(to_view_2[:, 1])))
    #plt.plot(np.mean(to_view_2[:, 3]), np.mean(to_view_2[:, 2]), 'v', color='r', label=r"BPC SET Average. $N_{SET}=$" + "{}".format(np.mean(to_view_2[:, 3])))
    plt.plot(np.mean(to_view_1[:, 5]), np.mean(to_view_1[:, 4]), 'o', color='r', label=r"PRET Average. $\bar{N}_{PRET}=$" + "{}".format(np.round(np.mean(to_view_1[:, 5]), 1)))
    plt.axhline(y = 0.02, color = 'r', label=r"$\epsilon_{max}=$ " + "{}".format(0.02), linestyle=(0, (1, 1)))
    plt.title(r"RCED Error $\epsilon$ vs. Bitstream length $N$ for 1000 ImageNet images")
    plt.xlabel(r"$N$")
    plt.ylabel(r"Error: $\epsilon$")
    plt.legend()
    plt.show()