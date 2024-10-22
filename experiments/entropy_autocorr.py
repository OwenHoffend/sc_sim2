import numpy as np
import matplotlib.pyplot as plt
from sim.datasets import dataset_sweep_2d
from sim.SNG import binomial_sng
from scipy.stats import entropy
from sim.PTM import get_vin_mc0, get_actual_vin

def test_lag_ptv():
    """The purpose of this function is to test the following:
        1. For a sequence like: 000...0111...1, the PTV with lag=1 should be about [0.5, 0, 0, 0.5]
        2. For a sequence like: 01010101010101, the PTV with lag=1 should be about [0, 0.5, 0.5, 0]
        3. For a random sequence with 0.5 probability, the PTV with lag=1 should be about [0.25, 0.25, 0.25, 0.25]

        Results: 
        1. Entropy is about 1
        2. Entropy is about 1
        3. Entropy is about 2
    """
    N = 2048
    seq1 = np.concatenate((np.zeros(int(N/2), dtype=np.bool_), np.ones(int(N/2), dtype=np.bool_)))
    ptv1 = get_actual_vin(seq1, lag=1)
    print(ptv1)
    print(entropy(ptv1, base=2))

    seq2 = np.zeros(N, dtype=np.bool_)
    seq2[::2] = 1
    ptv2 = get_actual_vin(seq2, lag=1)
    print(ptv2)
    print(entropy(ptv2, base=2))
    
    seq3 = binomial_sng(np.array([0.5,]), N)
    ptv3 = get_actual_vin(seq3, lag=1)
    print(ptv3)
    print(entropy(ptv3, base=2))

def entropy_autocorr_sim(circs):
    for circ in circs:
        N = 1024
        w = 1 #defines how much input entropy is actually available
        ds = np.linspace(0, 1, 1000) #dataset_sweep_2d(100, 100)
        Hins = []
        Hin_reals = []
        #Hin_t2s = []
        Hout_reals = []
        lag = 1
        for i, xs in enumerate(ds):
            print(i)
            xs = np.array([xs, 0.5]) #one of the two inputs is 0.5
            xs = circ.parr_mod(xs)

            v0 = get_vin_mc0(xs)
            Hin = entropy(v0, base=2)
            Hins.append(Hin)
            #print("xs: ({}, {}) has entropy Hin={}".format(xs[0], xs[1], Hin))

            #Hin_t2 = entropy(np.kron(v0, v0), base=2)
            #print("xs: ({}, {}) has entropy Hin_t2={}".format(xs[0], xs[1], Hin_t2))

            correct = circ.correct(xs)

            bs_mat = binomial_sng(xs, N)

            Hin_real = entropy(get_actual_vin(bs_mat), base=2) #do I need to divide by the lag component here?
            Hin_reals.append(Hin_real)
            #print("xs: ({}, {}) has actual entropy Hin_real={}".format(xs[0], xs[1], Hin_real))

            bs_out_sc = circ.run(bs_mat)
            sc_out = np.mean(bs_out_sc)

            v_out_l1 = get_actual_vin(np.expand_dims(bs_out_sc, axis=0), lag=lag)
            Hout_reals.append(entropy(v_out_l1, base=2))

            print("Err: ", np.abs(correct - sc_out))

        #plt.scatter(ds, Hin_reals, label="Input entropy")
        plt.scatter(ds, Hout_reals, label="output H for {} with mean: {}".format(circ.name, np.mean(Hout_reals)))
    plt.title("Input and output entropy for circuit: {}".format(circ.name))
    plt.legend()
    plt.show()