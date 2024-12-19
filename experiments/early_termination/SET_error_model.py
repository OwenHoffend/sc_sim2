import numpy as np
import matplotlib.pyplot as plt

from sim.sim import *
from sim.SNG import *
from sim.PCC import *
from sim.RNS import *
from sim.datasets import *
from sim.circs.circs import *

def hypergeo(N, p, Nmax):
    return (1/N) * p * (1-p) * (Nmax - N) / (Nmax - 1)

def binomial(N, p):
    return (1/N) * p * (1-p)

def fig_X():
    #Figure showing the error of a MUX up to Nmax for a uniform input dataset when n=3, w=2
    num = 10000
    w = 2
    circ = C_MUX_ADD(corr_inputs=False)
    sng = LFSR_SNG_WN(w, circ)
    ds = dataset_uniform(num, circ.nv)
    Nmax = circ.get_Nmax(w)
    Nrange = range(2, Nmax * 2 + 1)
    err_thresh = 0.15

    sim_result = sim_circ(sng, circ, ds, Nrange)

    #Nset calculation
    e_quants = np.abs(sim_result.trunc - sim_result.correct)
    quant_bias = np.mean(e_quants ** 2)
    target_var = err_thresh ** 2 - quant_bias
    mean_var = np.mean(sim_result.trunc * (1 - sim_result.trunc))
    Nset_hyper = Nmax * mean_var / (target_var * Nmax - target_var + mean_var)
    Nset_binom = mean_var / target_var

    errs = sim_result.RMSE_vs_N()

    hyper_model_errs = np.zeros((len(Nrange)))
    binom_model_errs = np.zeros((len(Nrange)))

    #model prediction:
    for i, xs in enumerate(ds):
        for j, N in enumerate(Nrange):
            z = sim_result.trunc[i]
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

    plt.plot(list(Nrange), errs, label="Actual error")
    plt.plot(list(Nrange), hyper_model_errs, label="Hypergeometric Model prediction")
    plt.plot(list(Nrange), binom_model_errs, label="Binomial Model prediction")
    plt.title(r"Error $\epsilon$ vs. Bitstream length $N$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"Error: $\epsilon$")

    e_quant_actual = errs[len(errs)-1]
    plt.axhline(y = err_thresh, color = 'r', label=r"$\epsilon_{max}=$ " + "{}".format(err_thresh), linestyle=(0, (1, 1)))
    plt.axhline(y = e_quant_actual, color = 'r', label=r"$\epsilon_{trunc}=$ " + "{}".format(e_quant_actual), linestyle=(0, (3, 1, 1, 1)))
    plt.axvline(x = Nmax, color = 'green', linestyle=(0, (1, 1)), label=r"$N_{max}$=" + "{}".format(Nmax))
    plt.axvline(x = 2*Nmax, color = 'limegreen', linestyle=(0, (3, 1, 1, 1)), label=r"$2N_{max}$="+ "{}".format(2*Nmax))
    plt.plot(Nset_hyper, err_thresh, 'o', color="red")
    plt.plot(Nset_binom, err_thresh, 'o', color="red")
    plt.legend()
    plt.show()