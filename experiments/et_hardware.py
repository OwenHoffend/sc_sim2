import numpy as np
from sim.Util import *
from sim.SCC import scc_mat
from sim.SNG import CAPE_sng, lfsr_sng
from sim.RNS import lfsr
from sim.PCC import CMP
from sim.ATPP import check_ATPP
from experiments.et_variance import get_dist
from sim.circs import robert_cross

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

def var_et(bs_out, max_var):
    m = bs_out.size
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
    print("VAR ET at : {} out of {}".format(N_et, m))
    return N_et, cnts

def RCED_et(max_precision, num_pxs, dist, staticN=None):
    #variance-based ET done using output samples only 
    nv = 4
    max_var = 0.01

    px_func = get_dist(dist, nv)
    ctr_sz = max_precision
    lfsr_width = ctr_sz + 1
    if staticN is not None:
        Nmax = staticN
    else:
        Nmax = 2 ** lfsr_width

    ets_var = []
    ets_cape = []
    mses_var = []
    mses_cape = []
    vars = []
    colors_var = []
    g = 0
    for _ in range(num_pxs):
        print("-----")

        #variance-based dynamic ET
        px = px_func()
        parr = np.concatenate((px, np.array([0.5,])))
        correct = 0.5 * (np.abs(px[0] - px[1]) + np.abs(px[2] - px[3]))
        cgroups = np.array([1, 1, 1, 1, 2])

        bs_mat = lfsr_sng(parr, Nmax, lfsr_width, cgroups=cgroups, pack=False)
        bs_out = np.zeros((Nmax,), dtype=np.bool_)
        for i in range(Nmax):
            bs_out[i] = robert_cross(*list(bs_mat[:, i]))

        pz = np.mean(bs_out)
        #full_length_mse = MSE(pz, correct)
        #print("full length mse: ", full_length_mse)
        N_et_var, cnts = var_et(bs_out, max_var)

        pz_et_var = np.mean(bs_out[:N_et_var])
        et_mse_var = MSE(pz_et_var, correct)
        mses_var.append(et_mse_var)
        print("et mse var: ", et_mse_var)

        ets_var.append(N_et_var)
        vars.append(pz * (1-pz))
        if et_mse_var > max_var:
            colors_var.append('r')
            g+=1
        else:
            colors_var.append('b')

        #CAPE-based dynamic ET
        bs_mat = CAPE_sng(parr, max_precision, cgroups, et=True, Nmax=Nmax)
        _, N_et_CAPE = bs_mat.shape
        ets_cape.append(N_et_CAPE)
        bs_out = np.zeros((N_et_CAPE,), dtype=np.bool_)
        for i in range(N_et_CAPE):
            bs_out[i] = robert_cross(*list(bs_mat[:, i]))
        pz = np.mean(bs_out)
        et_mse_cape = MSE(pz, correct)
        mses_cape.append(et_mse_cape)
        print("et mse CAPE: ", et_mse_cape)

        #plt.plot(cnts)
        #plt.title("Counter value vs. N for true output variance: {} \n Terminated at: {} out of {}".format(np.round(pz * (1-pz), 3), N_et, 512))
        #plt.ylabel("Counter value")
        #plt.xlabel("N")
        #plt.show()

    print("AVG ET var: ", np.mean(ets_var))
    print("AVG ET CAPE: ", np.mean(ets_cape))
    print("AVG MSE var: ", np.mean(mses_var))
    print("AVG MSE CAPE: ", np.mean(mses_cape))

    return np.mean(ets_var), np.mean(ets_cape), np.mean(mses_var), np.mean(mses_cape)

    #plt.scatter(vars, ets, c=colors_var)
    #plt.title("UNIFORM: ET N vs. var \n blue=within MSE bound, red=outside MSE bound")
    #plt.xlabel("Output variance")
    #plt.ylabel("Early termination N")
    #plt.show()
            
    #plt.hist(ets, 20)
    #plt.title("CENTER BETA: Hist of RCED Early Termination \n Avg length: {}, Percent >MSE thresh: {}%".format(np.mean(ets), 100* np.round(g/num_pxs, 2)))
    #plt.xlabel("Early termination N")
    #plt.ylabel("Frequency")
    #plt.show()

    