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

#Analysis for RET paper applications section
def ET_on_imagenet(idx, err_thresh, max_w, Nset_hyper=None, PRET_w=None):
    print("Image idx: ", idx)
    circ = C_RCED()

    ds = dataset_imagenet(2, mode='list', idxs=[idx])
    #ds.disp_img(0)

    #Van der Corput with Hypergeometric SET
    if Nset_hyper is None:
        Nset_hyper = SET_hyper(max_w, circ, ds, err_thresh)
    print("Nset_hyper: ", Nset_hyper)

    sng_vander = VAN_DER_CORPUT_SNG_WN(max_w, circ)
    sim_run_vander = sim_circ(sng_vander, circ, ds, Nset=np.round(Nset_hyper).astype(np.int32), loop_print=False)

    #BPC SET
    if PRET_w is None:
        N_PRET, PRET_err, PRET_w = analyze_PRET(max_w, circ, ds, err_thresh)
    print("PRET_w: ", PRET_w)

    sng_bpc = PRET_SNG_WN(PRET_w, circ, et=False)
    sim_run_bpc = sim_circ(sng_bpc, circ, ds, loop_print=False)
    #ds.disp_output_img(1.0 - sim_run_bpc.out, 0)

    #PRET
    sng_pret = PRET_SNG_WN(PRET_w, circ, lzd=True)
    sim_run_pret = sim_circ(sng_pret, circ, ds, loop_print=False)
    #ds.disp_output_img(1.0 - sim_run_pret.out, 0)

    print("IDX: {} ".format(idx) + "Vander + Hyper SET RMSE: ", sim_run_vander.RMSE())
    print("IDX: {} ".format(idx) + "Vander + Hyper SET avg N: ", sim_run_vander.avg_N())
    print("IDX: {} ".format(idx) + "BPC SET RMSE ", sim_run_bpc.RMSE())
    print("IDX: {} ".format(idx) + "BPC SET avg N", sim_run_bpc.avg_N())
    print("IDX: {} ".format(idx) + "PRET RMSE", sim_run_pret.RMSE())
    print("IDX: {} ".format(idx) + "PRET avg N", sim_run_pret.avg_N())

    #ds.disp_output_img(sim_run_pret.Ns, 0, scale=False, colorbar=True)

    return sim_run_vander.RMSE(), sim_run_vander.avg_N(), sim_run_bpc.RMSE(), sim_run_bpc.avg_N(), sim_run_pret.RMSE(), sim_run_pret.avg_N()

def ET_on_imagenet_mp():
    NUM_CORES = 10
    img_list = list(range(10))

    #SET on entire dataset
    ds = dataset_imagenet(2, mode='list', idxs=img_list)
    circ = C_RCED()
    err_thresh = 0.02
    max_w = 8
    Nset_hyper = SET_hyper(max_w, circ, ds, err_thresh)

    N_PRET, PRET_err, PRET_w = analyze_PRET(max_w, circ, ds, err_thresh)

    def f(i):
        return ET_on_imagenet(i, err_thresh, max_w, Nset_hyper=Nset_hyper, PRET_w=PRET_w)

    with Pool(NUM_CORES) as p:
        results = p.map(f, img_list)

    filename = "first_10.npy"
    arr = np.array(results)
    np.save("./results/rced_imagenet/set_entire_dataset/{}".format(filename), arr)

    #First run:
    #0 = (0.009998895606612577, 85.0, 0.010032511037475267, 64.0, 0.010032511037475267, 21.760022719235263)
    #1 = (0.01563836515145532, 33.0, 0.015117753900524378, 32.0, 0.015117753900524378, 5.608651037013787)

def plot_ET_on_imagenet_mp_results():
    #plotting/analysis function for ET on imagenet images
    
    #First load all of the daata
    first_1 = np.load("./results/rced_imagenet/set_every_image/first_1.npy")
    first_10 = np.load("./results/rced_imagenet/set_entire_dataset/first_10.npy")
    first_100 = np.load("./results/rced_imagenet/set_every_image/first_100.npy")
    next_100 = np.load("./results/rced_imagenet/set_every_image/next_100.npy")
    first_1000 = np.load("./results/rced_imagenet/set_every_image/first_1000.npy")

    #Result data
    #print("{}".format(np.mean(first_1, axis=0)))
    #[ 0.0099989  85.          0.01003251 64.          0.01003251 21.76002272]
    #print("{}".format(np.mean(first_10, axis=0)))
    #[ 0.0116108  67.5         0.01612983 38.4         0.01612983 18.2806746 ]
    #print("{}".format(np.mean(first_100, axis=0)))
    #[ 0.01095811 76.02        0.01664613 36.98        0.01664613 20.73495335]
    #print("{}".format(np.mean(first_1000, axis=0)))
    #[ 0.01130465 73.496       0.01669826 36.054       0.01669826 20.93796507]

    to_view = first_10

    plt.scatter(to_view[:, 1], to_view[:, 0], s=5, label="SET, Hypergeometric")
    plt.scatter(to_view[:, 5], to_view[:, 4], s=5, label="PRET")
    plt.plot(73.496, 0.01130465, 'o', color="red", label=r"SET Average. $N_{SET}=$" + "{}".format(73.50))
    plt.plot(20.937, 0.01669826, 'o', color="limegreen", label=r"PRET Average. $N_{RET}=$" + "{}".format(20.94))
    plt.axhline(y = 0.02, color = 'r', label=r"$\epsilon_{max}=$ " + "{}".format(0.02), linestyle=(0, (1, 1)))
    plt.title(r"RCED Error $\epsilon$ vs. Bitstream length $N$ for 1000 ImageNet images")
    plt.xlabel(r"$N$")
    plt.ylabel(r"Error: $\epsilon$")
    plt.legend()
    plt.show()