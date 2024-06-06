import numpy as np
import matplotlib.pyplot as plt

from sim.RNS import *
from sim.deterministic import *
from sim.SCC import *
from sim.PCC import *
from sim.SNG import *
from sim.Util import *
from experiments.early_termination.et_hardware import *

def ET_sim(circ, dataset, Nmax, w, max_var=0.001):
    SC_vals = []
    var_et_vals = []
    var_et_Ns = []
    cape_et_vals = []
    cape_et_Ns = []
    LD_et_vals = []
    LD_et_Ns = []
    for i, xs in enumerate(dataset):
        if i % 100 == 0:
            print("{} out of {}".format(i, dataset.shape[0]))

        xs = circ.parr_mod(xs) #Add constant inputs and/or duplicate certain inputs

        #Baseline LFSR-based SC performance
        bs_mat = lfsr_sng_efficient(xs, Nmax, w, cgroups=circ.cgroups, pack=False)
        bs_out_sc = circ.run(bs_mat)
        SC_vals.append(np.mean(bs_out_sc))

        #Variance-based ET performance
        N_et_var = var_et(bs_out_sc, max_var, exact=False)
        bs_out_var = bs_out_sc[:N_et_var]
        var_et_vals.append(np.mean(bs_out_var))
        var_et_Ns.append(N_et_var)

        #CAPE-without ET performance (Low discrepancy generator)
        #bs_mat = CAPE_sng(xs, w, circ.cgroups, et=False, Nmax=Nmax)
        #bs_out_LD = circ.run(bs_mat)
        #LD_et_vals.append(np.mean(bs_out_LD))
        #LD_et_Ns.append(bs_out_LD.size)

        #CAPE-based ET performance
        bs_mat = CAPE_sng(xs, w, circ.cgroups, et=True, Nmax=Nmax)
        bs_out_cape = circ.run(bs_mat)
        cape_et_vals.append(np.mean(bs_out_cape))
        cape_et_Ns.append(bs_out_cape.size)

    return SC_vals, var_et_vals, var_et_Ns, cape_et_vals, cape_et_Ns, LD_et_vals, LD_et_Ns

def ET_MSE_vc_N(circ, dataset, Nrange, w, max_var=0.001):
    """The primary function used to test RET schemes. Produces curves of MSE versus N for each proposed early termination method
        circ: The circuit which to run early termination on. Meant to be a circuit that inherits from sim.circs.Circ
        dataset: The dataset of Px values which to run the circuit with
        Nrange: A set of Nmax values to run the circuit at
        w: Baseline precision of the SC circuit
        max_var: Maximum variance for variance-based RET
    """

    correct_vals = []
    for xs in dataset:
        correct_vals.append(circ.correct(xs))
    correct_vals = np.array(correct_vals).flatten()
        
    SC_MSEs = []
    var_et_MSEs = []
    var_et_avg_Ns = []
    cape_et_MSEs = []
    cape_et_avg_Ns = []
    LD_et_MSEs = []
    LD_et_avg_Ns = []
    for Nmax in Nrange:
        SC_vals, var_et_vals, var_et_Ns, cape_et_vals, \
        cape_et_Ns, LD_et_vals, LD_et_Ns = \
        ET_sim(circ, dataset, Nmax, w, max_var=max_var)

        #MSEs
        sc_mse = MSE(correct_vals, np.array(SC_vals))
        SC_MSEs.append(sc_mse)
        var_mse = MSE(correct_vals, np.array(var_et_vals))
        var_et_MSEs.append(var_mse)
        LD_mse = MSE(correct_vals, np.array(LD_et_vals))
        LD_et_MSEs.append(LD_mse)
        cape_mse = MSE(correct_vals, np.array(cape_et_vals))
        cape_et_MSEs.append(cape_mse)

        #Ns
        var_Ns = np.mean(np.array(var_et_Ns))
        var_et_avg_Ns.append(var_Ns)
        LD_Ns = np.mean(np.array(LD_et_Ns))
        LD_et_avg_Ns.append(LD_Ns)
        cape_Ns = np.mean(np.array(cape_et_Ns))
        cape_et_avg_Ns.append(cape_Ns)

        print("SC: MSE: {}, N: {}".format(sc_mse, Nmax))
        print("CAPE: MSE: {}, N: {}".format(cape_mse, cape_Ns))
        print("LD: MSE: {}, N: {}".format(LD_mse, LD_Ns))
        print("VAR: MSE: {}, N: {}".format(var_mse, var_Ns))
    
    fig, ax = plt.subplots(1)
    plt.plot(Nrange, SC_MSEs, label="SC", marker='o')
    plt.plot(var_et_avg_Ns, var_et_MSEs, label="Var ET", marker='o')
    plt.plot(LD_et_avg_Ns, LD_et_MSEs, label="LD ET", marker='o')
    plt.plot(cape_et_avg_Ns, cape_et_MSEs, label="Precision ET", marker='o')
    plt.axhline(y = max_var, color='purple', linestyle = '--', label="Max MSE")
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.legend()
    plt.xlabel("Bitstream length (N)")
    plt.ylabel("MSE")
    plt.title("MSE vs. Bitstream length for circuit: {}".format(circ.name))
    plt.show()

def static_ET(circ, dataset, w, max_var=0.001, plot=True):
    N_ets = []
    Nmax = circ.get_Nmax(w)
    pzs = []
    print("Nmax: ", Nmax)
    for xs in dataset:
        pz = circ.correct(xs)
        pzs.append(pz)
        N_et = (Nmax * pz * (1-pz)) / (max_var * Nmax - max_var + pz * (1-pz))
        N_ets.append(N_et)

    avg_N_et = np.mean(N_ets)
    median_N_et = np.median(N_ets)
    print("Avg N: ", avg_N_et)
    print("Max N: ", np.max(N_ets))

    static_ET = np.mean(N_ets)

    correct_vals = []
    for xs in dataset:
        correct_vals.append(circ.correct(xs))
    correct_vals = np.array(correct_vals).flatten()
        
    Nrange = []
    i = 1
    while 2 ** i <= 256:
        Nrange.append(2 ** i)
        i += 1

    SC_MSEs = []
    for Nmax in Nrange:
        SC_vals = []
        for i, xs in enumerate(dataset):
            print("{} out of {}".format(i, dataset.shape[0]))
            xs = circ.parr_mod(xs) #Add constant inputs and/or duplicate certain inputs

            #Baseline LFSR-based SC performance
            bs_mat = lfsr_sng_efficient(xs, Nmax, w, cgroups=circ.cgroups, pack=False)
            bs_out_sc = circ.run(bs_mat)
            SC_vals.append(np.mean(bs_out_sc))
        sc_mse = MSE(correct_vals, np.array(SC_vals))
        SC_MSEs.append(sc_mse)

    if plot:
        fig, ax1 = plt.subplots()
        ax1.axvline(x = static_ET, color='purple', linestyle = '--', label="avg")
        ax1.axvline(x = median_N_et, color='red', linestyle = '--', label="med")
        ax2 = ax1.twinx()
        ax2.plot(Nrange, SC_MSEs, color='y')
        ax1.hist(N_ets, bins="auto")
        ax1.set_xlabel("ET length")
        ax1.set_ylabel("Frequency")
        ax2.set_ylabel("MSE")
        plt.title("Static ET: {} \n median: {}".format(static_ET, median_N_et))
        ax1.legend(loc="upper center")
        plt.show()

    return np.ceil(static_ET).astype(np.int32)

def partial_bitstream_value_plot(bss, ps):
    #Takes a list of bitstreams. For each, plot the estimated value of the bitstream at each time step
    for bs_idx, bs in enumerate(bss):
        p_vals = []
        running_total = 0
        for idx, b in enumerate(bs):
            running_total += b
            p_vals.append(running_total/(idx+1))
        plt.plot(p_vals)
    plt.ylabel("Partial Bitstream Value")
    plt.xlabel("Bitstream Length")
    plt.title("""Bitstream value vs. Partial Bitstream Length""")
    plt.show()

def et_plot_1d(rns, w):
    N = 2 ** w
    rand = bit_vec_arr_to_int(rns(w, N))
    y = np.zeros_like(rand)
    c = np.array([[int(x*255.0/N), int(x*255.0/N), 255] for x in range(N)])
    plt.scatter(rand, y, s = 10, c = c/255.0)
    plt.show()

def et_plot_2d(rns, tile_func, w):
    N = 2 ** w #generate full period
    full_N = N ** 2
    
    x_rns, y_rns = tile_func(rns, w, N)

    c = np.array([[int(x*255.0/full_N), int(x*255.0/full_N), 255] for x in range(full_N)])
    plt.scatter(x_rns, y_rns, s = 10, c = c/255.0)
    plt.xlabel("X RNS value")
    plt.ylabel("Y RNS value")
    plt.title("{} : {}".format(rns.__name__, tile_func.__name__))
    plt.savefig('./experiments/plots/{}_{}_points.png'.format(rns.__name__, tile_func.__name__))
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

def disc_plot_2d(rns, tile_func, w):
    N = 2 ** w #generate full period
    
    precision_points = []
    i = 1
    for _ in range(2*w):
        precision_points.append(i)
        i *= 2

    x_rns, y_rns = tile_func(rns, w, N)
    discs = []
    S = []
    for i in precision_points:
        print(i)
        S.append([x_rns[i]/N, y_rns[i]/N])
        P = get_possible_Ps(i)
        discs.append(star_disc_2d(np.array(S), P))
    return discs

    #plt.plot(discs, marker='o')
    #plt.xlabel("SN Length (log2 scale)")
    #plt.ylabel("Discrepancy")
    #plt.title("{} : {}".format(rns.__name__, tile_func.__name__))
    #plt.show()

def scc_vs_ne_common(bs, N, name):
    cs = np.empty(N**2)
    pow2s = []
    for i in range(N**2):
        c = scc(bs[0, :i], bs[1,:i])
        if np.isnan(c):
            c = 0.0
        cs[i] = c 
        if np.log2(i) == np.ceil(np.log2(i)):
            pow2s.append(i) #bad code but I'm lazy
            print("i: {}, c: {}".format(i, cs[i]))
    print("Mean SCC: ", np.mean(np.abs(cs)))
    print("Std. SCC: ", np.std(np.abs(cs)))

    plt.plot(cs)
    plt.scatter(pow2s, cs[pow2s], color="red")
    plt.scatter(N ** 2, cs[-1], color="red") #add the last point
    plt.xlabel("SN Length")
    plt.ylabel("SCC")
    plt.title("SCC vs Early Termination: {}".format(name))
    plt.show()

def scc_vs_ne_others(parr, rns, tile_func, w):
    """Plot the SCC with respect to early termination point"""
    N = 2 ** w #generate full period
    x_rns, y_rns = tile_func(rns, w, N)
    bs = sng_from_pointcloud(parr, np.stack((x_rns, y_rns)), pack=False)
    scc_vs_ne_common(bs, N, "{} : {}".format(rns.__name__, tile_func.__name__))

def scc_vs_ne_CAPE(parr, w):
    N = 2 ** w #generate full period
    bs = CAPE_sng(parr, N**2, w, pack=False)
    scc_vs_ne_common(bs, N, "CAPE")

def scc_vs_ne_SA(px, py, tile, w):
    N = 2 ** w
    bsx = SA_sng(px, N, w, pack=False)
    bsy = SA_sng(py, N, w, pack=False)
    bsx_r, bsy_r = tile(bsx, bsy, N)
    bs = np.stack((bsx_r, bsy_r))
    scc_vs_ne_common(bs, N, "Streaming Accurate SNG, {}".format(tile.__name__))

def check_ATPP(w, sng):
    N = 2 ** w
    ps = get_possible_Ps(N)
    for w_t_p in range(1, w):
        for p in ps:
            if p == 1:
                continue
            bs = sng(np.array((p,)), N, w, pack=False)[0, :2**w_t_p]
            #bs = SA_SNG(p, N, pack=False)[:2**w_t_p]
            px = np.mean(bs)
            p_trunc = np.floor(p * 2 ** w_t_p) / (2 ** w_t_p)
            if px != p_trunc:
                return False
    return True

def get_progressive_SCCs(w, sng, tile, all_vals=False):
    N = 2 ** w
    ps = get_possible_Ps(N)
    for px in ps:
        if px == 1:
            continue
        for py in ps:
            if py == 1:
                continue
            if sng.__name__ == "CAPE_sng":
                bs_mat = CAPE_sng(np.array((px, py)), N**2, 2*w, pack=False)
                bsx_r, bsy_r = bs_mat[0, :], bs_mat[1, :]
            else:
                bs_mat = sng(np.array([px, py]), N, w, pack=False)
                bsx_r, bsy_r = tile(bs_mat[0, :], bs_mat[1, :], N)
            if all_vals:
                for i in range(1, N ** 2):
                    yield scc(bsx_r[:i], bsy_r[:i])
            else:
                for w_t_p in range(1, 2*w):
                    yield scc(bsx_r[:2**w_t_p], bsy_r[:2**w_t_p])

def get_progressive_PCorrs_pointcloud(w, rns, tile_func):
    N = 2 ** w
    x_rns, y_rns = tile_func(rns, w, N)
    S = np.empty((N ** 2, 2))
    for i, j in zip(x_rns, y_rns):
        S[i, 0] = i 
        S[i, 1] = j
    for i in range(1, N ** 2):
        yield np.corrcoef(x_rns[:i], y_rns[:i])[1, 0]

def check_MATPP(w, sng, tile):
    #seems like clock division preserves correlation
    #rotation does not
    for c in get_progressive_SCCs(w, sng, tile):
        if c != 0:
            return False
    return True

def et_plot_multi(w):
    funcs = [lfsr, true_rand, counter, van_der_corput]
    tile_methods = [full_width_2d, rotation_2d, clock_division_2d]
    markers = ['o', 'v', '*']
    for func in funcs:
        for i, method in enumerate(tile_methods):
            et_plot_2d(func, method, w)
            #discs = disc_plot_2d(func, method, w)
            #plt.plot(discs, marker=markers[i], label=method.__name__)
        #plt.xlabel("SN Length (log2 scale)")
        #plt.ylabel("Discrepancy")
        #plt.title("{}_disc".format(func.__name__))
        #plt.legend()
        #plt.savefig('./experiments/plots/{}_disc.png'.format(func.__name__))
        #plt.figure().clear()
        #plt.close()
        #plt.cla()
        #plt.clf()
        #plt.show()

def plot_SCC_avg_vs_ne(w):
    #Unlike scc_vs_ne above, this function will take the average over all possible probability values
    #This average plot will also be compared against the Pearson correlation coefficient of the point cloud to see how well it corresponds
    sngs = [counter_sng]
    rnses = [counter]

    tiles = [rotation_2d_from_bs]
    tiles_rns = [rotation_2d]
    use_pearson = True #compare against evaluating the Pearson correlation of the point cloud 

    sccs = []
    for sng in sngs:
        #print("{}: ATPP: {}".format(sng.__name__, check_ATPP(7, sng)))
        for tile in tiles:
            #sccs = np.array(list(get_progressive_SCCs(w, sng, tile, all_vals=True)))
            #sccs = sccs.reshape((((2 ** w - 1) ** 2), (2 ** (2*w) - 1)))
            #msccs = np.mean(np.abs(sccs), axis=0)
            #np.save("./data/{}_{}_sccs.npy".format(sng.__name__, tile.__name__), msccs)

            sccs = np.load("./data/{}_{}_sccs.npy".format(sng.__name__, tile.__name__))
            plt.plot(sccs, label="{}".format(sng.__name__))

    pow2s = np.array([2 ** i - 1 for i in range(2*w)])
    plt.scatter(pow2s, sccs[pow2s], color="red")
    if use_pearson:
        for rns in rnses:
            for tile in tiles_rns:
                #sccs = np.abs(np.array(list(get_progressive_PCorrs_pointcloud(w, rns, tile))))
                #np.save("./data/{}_{}_sccs_pearson.npy".format(rns.__name__, tile.__name__), sccs)

                sccs = np.load("./data/{}_{}_sccs_pearson.npy".format(rns.__name__, tile.__name__))
                plt.plot(sccs, label="{}_pearson".format(rns.__name__))

    plt.scatter(np.array([2 ** (2*w)]), sccs[-1], color="red")
    plt.title("Avg. abs(pearson) vs Early Termination: Counter")
    plt.xlabel("SN Length")
    plt.ylabel("Avg. abs(SCC)")
    plt.legend()
    plt.show()

def et_mul3_uniform():
    max_var = 0.01
    num_tests = 100
    avg_N_var = 0.0
    avg_N_CAPE = 0.0

    correct_arr = np.zeros((num_tests,))
    sc_full = np.zeros((num_tests,))
    var_et = np.zeros((num_tests,))
    cape_et = np.zeros((num_tests,))

    for max_precision in [4, ]:
        for i in range(num_tests):
            px = np.random.uniform(size=(3,))
            correct, pz_full, pz_et_var, pz_et_both, \
            N_et_var, pz_et_CAPE, N_et_CAPE, N_et_both = \
                mul3_et_kernel(px, max_precision, max_var)
            correct_arr[i] = correct
            sc_full[i] = pz_full
            var_et[i] = pz_et_var
            cape_et[i] = pz_et_CAPE
            avg_N_var += N_et_var
            avg_N_CAPE += N_et_CAPE

        avg_N_var /= num_tests
        avg_N_CAPE /= num_tests
        print("max prec: ", max_precision)
        print("avg N var: ", avg_N_var)
        print("avg N CAPE: ", avg_N_CAPE)
        print("MSE SC full: ", MSE(correct_arr, sc_full))
        print("MSE var: ", MSE(correct_arr, var_et))
        print("MSE CAPE: ", MSE(correct_arr, cape_et))

    #get the number N values, then re-run
    #for staticN in ...