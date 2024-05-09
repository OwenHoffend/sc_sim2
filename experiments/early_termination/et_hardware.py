import numpy as np
import time
from sim.Util import *
from sim.SCC import scc_mat
from sim.SNG import *
from sim.PCC import CMP
from sim.ATPP import check_ATPP
from experiments.early_termination.et_variance import get_dist

def CAPE_basic_test():
    w = 3
    n = 4
    parr = np.random.uniform(size=(n,))
    N = 2 ** (w * n)
    Bx = parr_bin(parr, w, lsb="right")

    trunc = fp_array(Bx)

    input_stream = CAPE_sng(parr, N, w, pack=False)
    input_stream_ET = CAPE_sng(parr, N, w, pack=False, et=True)
    assert np.all(np.isclose(np.mean(input_stream, axis=1), np.mean(input_stream_ET, axis=1)))

    normal_and = np.bitwise_and(input_stream[0, :], input_stream[1, :])
    et_and = np.bitwise_and(input_stream_ET[0, :], input_stream_ET[1, :])

    assert trunc[0] * trunc[1] == np.mean(normal_and) == np.mean(et_and)

    print('Bx', Bx)
    print('reduction ratio', input_stream.shape[1] / input_stream_ET.shape[1])
    print('correct: ', trunc[0] * trunc[1])
    print('normal', np.mean(normal_and))
    print('et', np.mean(et_and))

    #confirmed that the bypass-counter design works

def CAPE_based_ET_stats(n, w, dist, num_pxs):
    N = 2 ** (w * n)
    px_func = get_dist(dist, n)
    Ns = []
    for _ in range(num_pxs):
        parr = px_func()
        Bx = parr_bin(parr, w, lsb="right")

        #compute the bypass bit vector
        #Trailing zero detection
        tzd = np.zeros((n, w), dtype=np.bool_)
        col = np.zeros((n, ), dtype=np.bool_)
        for i in reversed(range(w)):
            col = np.bitwise_or(Bx[: , i], col)
            tzd[:, i] = np.bitwise_not(col)

        #reorder to correspond to CAPE counter bits
        tzd = np.flip(tzd, axis=1)
        bp = tzd.reshape((n * w), order='F') #corresponds to a column-major ordering. F stands for Fortran *shrug*

        ctr_width = n * w - np.sum(bp)
        N_new = np.minimum(N, 2 ** ctr_width)
        Ns.append(N_new)
    Navg = np.mean(np.array(Ns))
    return Navg / N

def CAPE_corr_basic_test():
    precision = 5
    nv = 4
    p = np.random.uniform(size=nv)
    Bx = parr_bin(p, precision, lsb='right')
    ctr = 0
    input_stream = np.zeros((nv, 2 ** precision), dtype=np.bool_)
    for i in range(2 ** precision):
        ctr_bin = bin_array(ctr, precision)
        for j in range(nv):
            input_stream[j, i] = CMP(np.flip(ctr_bin), Bx[j, :])
        ctr += 1
    check_ATPP(input_stream, Bx)
    print(scc_mat(input_stream))

def CAPE_test():
    dists = ["uniform", "MNIST_beta", "center_beta"]
    ns = [1, 2, 3, 4, 5, 6, 7, 8]
    ws = [1, 2, 3, 4, 5, 6]

    #results = np.empty((3, len(ns), len(ws)), dtype=object)
    #for i, dist in enumerate(dists):
    #    for j, n in enumerate(ns):
    #        for k, w in enumerate(ws):
    #            if n * w >= 32:
    #                continue
    #            results[i, j, k] = CAPE_based_ET_stats(n, w, dist, 10000)
    #np.save("cape_ET.npy", results)

    results = np.load("cape_ET.npy", allow_pickle=True)

    for d in range(3):
        for i in range(len(ws)):
            plt.plot(ns, results[d, :, i], label="{}-bits of precision".format(i+1))
        plt.title("Relative bitstream length, {} \n (Lower is better)".format(dists[d]))
        plt.xlabel("Number of inputs (n)")
        plt.ylabel("Relative bitstream length")
        plt.legend()
        plt.show()

def CAPE_vs_var(num_pxs):
    w = 4
    n = 2
    N = 2 ** (w * n) #before any early termination
    ets_CAPE = []
    ets_var = []
    for _ in range(num_pxs):
        parr = np.array([np.random.uniform(), np.random.uniform()])
        Bx = parr_bin(parr, w, lsb="right")

        trunc = fp_array(Bx)
        correct = trunc[0] * trunc[1]

        #CAPE-based approach
        input_stream_ET = CAPE_sng(parr, N, w, pack=False, et=True)
        N_et_CAPE = input_stream_ET[0, :].size
        CAPE_et = np.bitwise_and(input_stream_ET[0, :], input_stream_ET[1, :])
        assert correct == np.mean(CAPE_et)

        #variance-based approach
        lfsr_bs = lfsr_sng(parr, N, w, pack=False)
        bs_out = np.bitwise_and(lfsr_bs[0, :], lfsr_bs[1, :])
        N_et_var, _ = var_et(bs_out, 0.01)

        #print("CAPE: ", N_et_CAPE)
        #print("Var: ", N_et_var)
        pz_var = np.mean(bs_out[:N_et_var])
        #print("Var MSE: ", MSE(correct, pz_var))

        ets_CAPE.append(N_et_CAPE / N)
        ets_var.append(N_et_var / N)
    print(np.mean(np.array(ets_CAPE)))
    print(np.mean(np.array(ets_var)))
    plt.hist(ets_CAPE, 20, label="CAPE")
    plt.hist(ets_var, 20, label="variance")
    plt.title("CAPE vs Var ET")
    plt.xlabel("Early termination N")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def var_et(bs_out, max_var, Nprec=None, power_of_2=False):
    if Nprec is not None:
        m = min(Nprec, bs_out.size)
    else:
        m = bs_out.size

    if m == 0:
        return 1, []
    
    pz = np.mean(bs_out)

    #dynamic ET hardware
    var = np.bitwise_and(bs_out[1:], np.bitwise_not(bs_out[:-1]))
    #print("var est: " , np.mean(var))
    #print("actual var: ", pz * (1-pz))

    N_et = m
    N_min = np.rint(m / (4*(max_var*m - max_var) + 1)).astype(np.int32) #1.0 / (4 * max_var)
    ell = clog2(N_min)
    cnt = N_min
    cnts = []
    for i in range(m-1):
        if var[i] and cnt < 2 ** ell - 1:
            cnt += 3
        elif not var[i] and cnt > 0:
            cnt -= 1
        if cnt == 0 and i < N_et:
            N_et = i
        cnts.append(cnt)
    #print("VAR ET at : {} out of {}".format(N_et, m))
    if power_of_2:
        N_et = 2 ** clog2(N_et)
    return N_et, cnts

def CAPE_N_analytical_1input():
    """Analytical calculation of the early-terminated bitstream length for CAPE, for a circuit with 1 input"""

    max_w = 8
    for n in range(1, 5):
        bseqs = np.array([mbseq(w, n) for w in range(max_w)])
        print(bseqs[max_w-1])
        plt.plot(bseqs, label="n: {}".format(n))
    plt.title("CAPE bitstream savings ratio vs w and n")
    plt.xlabel("Binary bit-width w")
    plt.ylabel("Savings ratio: $N_{et}/N$")
    plt.legend()
    plt.show()