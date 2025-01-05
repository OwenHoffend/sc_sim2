import numpy as np
import matplotlib.pyplot as plt

from sim.sim import *
from sim.SNG import *
from sim.PCC import *
from sim.RNS import *
from sim.datasets import *
from sim.circs.circs import *
from experiments.early_termination.RET import *

def hypergeo(N, p, Nmax):
    return (1/N) * p * (1-p) * (Nmax - N) / (Nmax - 1)

def binomial(N, p):
    return (1/N) * p * (1-p)

def fig_X():
    #Figure showing the error of a MUX up to Nmax for a uniform input dataset when n=3, w=2
    num = 1000
    err_thresh = 0.02
    w = 8

    #Circs
    #circ = C_MUX_ADD(corr_inputs=False)
    circ = C_RCED()
    #circ = C_Gamma() #Causes Nmax to be really huge

    #SNGs
    #sng = HYPER_SNG_WN(w, circ)
    #sng = LFSR_SNG_WN(w, circ)
    #sng = COUNTER_SNG_WN(w, circ)
    sng = VAN_DER_CORPUT_SNG_WN(w, circ)
    
    #conventional simulation
    #ds = dataset_uniform(num, circ.nv)
    ds = dataset_imagenet(2, windows_per_img=100, num_imgs=10)
    Nmax = circ.get_Nmax(w)
    Nrange = list(range(2, Nmax * 2 + 1))
    sim_run = sim_circ_NSweep(sng, circ, ds, Nrange)

    #Nset calculation
    e_quants = np.abs(sim_run.trunc - sim_run.correct)
    quant_bias = np.mean(e_quants ** 2)
    target_var = err_thresh ** 2 - quant_bias
    mean_var = np.mean(sim_run.trunc * (1 - sim_run.trunc))
    Nset_hyper = Nmax * mean_var / (target_var * Nmax - target_var + mean_var)
    Nset_binom = mean_var / target_var

    #N_PRET w calculation
    N_PRET, PRET_err, PRET_w = analyze_PRET(w, circ, ds, err_thresh)

    #PRET simulation
    sng_pret = PRET_SNG_WN(PRET_w, circ, lzd = True)
    sim_run_pret = sim_circ(sng_pret, circ, ds)
    Ns, errs = sim_run.RMSE_vs_N()
    Ns_pret, errs_pret = sim_run_pret.RMSE_vs_N()

    print("RET w: ", PRET_w)
    print("PRET avg N: ", sim_run_pret.avg_N())
    print("PRET avg err: ", sim_run_pret.RMSE())

    hyper_model_errs = np.zeros((len(Nrange)))
    binom_model_errs = np.zeros((len(Nrange)))

    #binomial/hypergeometric model prediction:3
    for i, xs in enumerate(ds):
        for j, N in enumerate(Nrange):
            z = sim_run.trunc[i]
            if j < Nmax:
                hvar = hypergeo(N, z, Nmax)
                hyper_model_errs[j] += hvar
            else:
                hyper_model_errs[j] = np.nan
            bvar = binomial(N, z)
            binom_model_errs[j] += bvar

    hyper_model_errs = np.sqrt(hyper_model_errs / num + quant_bias)
    binom_model_errs = np.sqrt(binom_model_errs / num + quant_bias)

    print("Nset_hyper: ", Nset_hyper)
    print("Nset_binom: ", Nset_binom)
    print("N_PRET simulated N: ", N_PRET)
    print("N_PRET simulated err: ", PRET_err)

    plt.plot(list(Nrange), errs, label="Actual error")
    plt.plot(list(Nrange), hyper_model_errs, label="Hypergeometric Model prediction")
    plt.plot(list(Nrange), binom_model_errs, label="Binomial Model prediction")
    plt.plot(N_PRET, PRET_err, 'o', color='limegreen', label=r"$N_{PRET} $ without LZD: " + "{}".format(np.round(N_PRET)))
    plt.plot(sim_run_pret.avg_N(), sim_run_pret.RMSE(), 'o', color='lightgreen', label=r"$N_{PRET}$ with LZD: " + "{}".format(np.round(sim_run_pret.avg_N())))
    plt.title(r"Error $\epsilon$ vs. Bitstream length $N$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"Error: $\epsilon$")

    e_quant_actual = errs[len(errs)-1]
    plt.axhline(y = err_thresh, color = 'r', label=r"$\epsilon_{max}=$ " + "{}".format(err_thresh), linestyle=(0, (1, 1)))
    plt.axhline(y = e_quant_actual, color = 'r', label=r"$\epsilon_{trunc}=$ " + "{}".format(np.round(e_quant_actual, 3)), linestyle=(0, (3, 1, 1, 1)))
    plt.axvline(x = Nmax, color = 'green', linestyle=(0, (1, 1)))
    plt.axvline(x = 2*Nmax, color = 'limegreen', linestyle=(0, (3, 1, 1, 1)))
    plt.plot(Nset_hyper, err_thresh, 'o', color="red", label=r"$N_{SET}$, hyper: " + "{}".format(np.round(Nset_hyper, 3)))
    plt.plot(Nset_binom, err_thresh, 'o', color="blue", label=r"$N_{SET}$, binom: " + "{}".format(np.round(Nset_binom, 3)))
    plt.legend()
    plt.show()