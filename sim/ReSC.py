import numpy as np
from scipy.special import comb
from sim.Util import *
from sim.RNS import *
from sim.PCC import *
from sim.SCC import *
from sim.SNG import CAPE_sng
from experiments.et_hardware import var_et

def ReSC(inputs):
    w, N = inputs.shape 
    sz = int((w-1) / 2)
    x = inputs[:sz, :]
    b = inputs[sz:, :]
    sel = np.sum(x, axis=0)
    z = np.take_along_axis(b, sel.reshape(1, N), axis=0)
    return z

def bernstein(coeffs, x):
    def B_i_n(i, n, x_):
        return comb(n, i) * (x_**i) * (1-x_)**(n-i)
    out = np.zeros_like(x)
    n = coeffs.size
    for i in range(n):
        out += coeffs[i] * B_i_n(i, n-1, x)
    return out

def gamma_correction():
    num_samples = 25
    x_vals = np.linspace(0, 1, num_samples)

    #Bernstein approximation from "An Architecture for Fault-Tolerant Computation with Stochastic Logic"
    b = [
        0.0955,
        0.7207,
        0.3476,
        0.9988,
        0.7017,
        0.9695,
        0.9939
    ]
    w = 8

    #bbin = parr_bin(np.array(b), w, lsb="left")
    y_vals = np.zeros((num_samples,))
    y_vals_et = np.zeros((num_samples), )
    y_vals_CAPE_et = np.zeros((num_samples), )
    savings = []
    savings_CAPE = []
    for idx, x in enumerate(x_vals):

        print(idx)
        Nmax = 1024

        #Variance-based early termination:
        #Get the input bitstreams

        parr = np.array([x,] * 6 + b)
        cgroups = np.array([1, 2, 3, 4, 5, 6] + [7 for _ in range(7)])
        bs_mat = lfsr_sng(parr, Nmax, w, cgroups=cgroups, pack=False)
        bs_out = ReSC(bs_mat).flatten()

        N_et_var, _ = var_et(bs_out, 0.001)
        savings.append(N_et_var)
        y_vals[idx] = np.mean(bs_out)
        y_vals_et[idx] = np.mean(bs_out[:N_et_var])

        #CAPE-based early termination:
        bs_mat = CAPE_sng(parr, w, cgroups, Nmax=Nmax, et=True)
        #print(scc_mat(bs_mat))
        bs_out = ReSC(bs_mat).flatten()
        y_vals_CAPE_et[idx] = np.mean(bs_out)
        savings_CAPE.append(bs_out.size)
        
    plt.title("Gamma correction Early Termination Test \n Avg. var et length: {}/1024 \n Avg. CAPE et length: {}"
              .format(np.mean(np.array(savings)), np.mean(np.array(savings_CAPE))))
    plt.plot(x_vals, x_vals ** 0.45, label="correct")
    plt.plot(x_vals, bernstein(np.array(b), x_vals), label="bernstein approx")
    #plt.plot(x_vals, y_vals, label="SC, N=1024")
    plt.plot(x_vals, y_vals_et, label="SC, var-based ET")
    plt.plot(x_vals, y_vals_CAPE_et, label="SC, CAPE-based ET")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    print(MSE(y_vals, x_vals ** 0.45))
    #print(MSE(y_vals_et, x_vals ** 0.45))
    print(MSE(y_vals_CAPE_et, x_vals ** 0.45))

    plt.plot(x_vals, savings)
    plt.plot(x_vals, savings_CAPE)
    plt.title("Actual bits sampled for each X value")
    plt.show()

def ReSC_test():
    inputs = np.array([
        [True, False],
        [True, False],
        [True, True],

        [False, False],
        [True, True],
        [False, True],
        [True, False]
    ])
    ReSC(inputs)