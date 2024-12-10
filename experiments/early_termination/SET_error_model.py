import numpy as np
import matplotlib.pyplot as plt

from sim.circs.circs import *
from sim.sim import *

def fig_X(circ: Circ, ds):
    #Figure showing the error of a MUX up to Nmax for a uniform input dataset when n=3, w=2
    num = 1000
    w = 3

    Nmax = 2 ** (w * circ.n)

    #SET calculation
    err_thresh = 0.075
    corr_vals = gen_correct(circ, ds) #Z*
    trunc_vals = gen_correct(circ, ds, trunc_w=w) #Z* truncated to w
    e_quants = np.abs(trunc_vals-corr_vals)
    quant_bias = np.mean(e_quants ** 2)

    #Nset calculation
    target_var = err_thresh ** 2 - quant_bias
    mean_var = np.mean(trunc_vals * (1 - trunc_vals))
    Nset_hyper = Nmax * mean_var / (target_var * Nmax - target_var + mean_var)
    Nset_binom = mean_var / target_var

    Nrange = range(2, 2*Nmax+1)
    errs = np.zeros((len(Nrange)))
    hyper_model_errs = np.zeros((len(Nrange)))
    binom_model_errs = np.zeros((len(Nrange)))
    for i, xs in enumerate(ds):
        print(i)
        xs = circ.parr_mod(xs)
        
        #RNS Choice:
        bs_mat_full = true_rand_precise_sample(xs, w)
        #bs_mat_full = lfsr_sng_precise_sample(xs, w)
        
        bs_mat_full = np.concatenate((bs_mat_full, bs_mat_full), axis=1)
        for j, N in enumerate(Nrange):
            bs_mat = bs_mat_full[:, :N]
            bs_out = circ.run(bs_mat)
            out_val = np.mean(bs_out)
            correct = circ.correct(xs)
            errs[j] += MSE(out_val, correct)
            
            #model prediction:
            if j < Nmax:
                z = trunc_vals[i]
                hvar = hypergeo(N, z, Nmax)
                bvar = binomial(N, z)

                hyper_model_errs[j] += hvar
                binom_model_errs[j] += bvar
            else:
                hyper_model_errs[j] = np.nan
                binom_model_errs[j] = np.nan

    errs = np.sqrt(errs / num)
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