import numpy as np
import matplotlib.pyplot as plt

from sim.RNS import *
from sim.circs.circs import *
from sim.datasets import *
from sim.deterministic import *
from sim.SCC import *
from sim.PCC import *
from sim.SNG import *
from sim.Util import *
from experiments.early_termination.old_earlytermination.SET import *
from experiments.early_termination.old_earlytermination.et_sim import *
from experiments.discrepancy import *
from experiments.early_termination.old_earlytermination.et_hardware import *

import numpy as np
from pylfsr import LFSR
from sim.Util import bin_array, int_array

fpoly_cache = {}
def lfsr(w, N, poly_idx=0, use_rand_init=True):
    """
    w is the bit-width of the generator (this is a SINGLE RNS)
    N is the length of the sequence to sample (We could be sampling less than the full period of 2 ** w)
    """
    cache_str = str(w) + ":" + str(poly_idx)
    if cache_str in fpoly_cache: #this optimization greatly speeds up the lfsr code :)
        fpoly = fpoly_cache[cache_str]
    else:
        fpoly = LFSR().get_fpolyList(m=int(w))[poly_idx]
        fpoly_cache[cache_str] = fpoly
        
    all_zeros = np.zeros(w)
    while True:
        zero_state = np.random.randint(2, size=w) #Randomly decide where to put the zero state
        if not np.all(zero_state == all_zeros):
            break

    if use_rand_init:
        while True:
            init_state = np.random.randint(2, size=w) #Randomly pick an init state
            if not np.all(init_state == all_zeros):
                break
    else:
        init_state = np.zeros((w,))
        init_state[0] = 1

    L = LFSR(fpoly=fpoly, initstate=init_state)

    lfsr_bits = np.zeros((w, N), dtype=np.bool_)
    last_was_zero = False
    for i in range(N):
        if not last_was_zero and \
            np.all(L.state == zero_state):
                lfsr_bits[:, i] = all_zeros
                last_was_zero = True
                continue
        last_was_zero = False
        L.runKCycle(1)
        lfsr_bits[:, i] = L.state
    return lfsr_bits

def lfsr_sng_efficient(parr, N, w, corr=0, cgroups=None, pack=True):
    n = parr.size
    pbin = parr_bin(parr, w, lsb="left")
    pbin_ints = int_array(pbin)
    
    #Generate the random bits
    bs_mat = np.zeros((n, N), dtype=np.bool_)
    r = lfsr(w, N)
    r_ints = int_array(r.T)

    if cgroups is not None:
        g = cgroups[0]
    for i in range(n):
        if cgroups is not None:
            if cgroups[i] != g:
                g = cgroups[i]
                r = lfsr(w, N, poly_idx=g)
                r_ints = int_array(r.T)
        elif not corr: #if not correlated, get a new independent rns sequence
            r = lfsr(w, N)
            r_ints = int_array(r.T)

        p = pbin_ints[i]
        for j in range(N):
            bs_mat[i, j] = p > r_ints[j]

    return sng_pack(bs_mat, pack, n)

def ET_MSE_vc_N(ds, circ, e_min, e_max, SET_override=None):
    """The primary function used to test RET schemes. Produces curves of MSE versus N for each proposed early termination method
        circ: The circuit which to run early termination on. Meant to be a circuit that inherits from sim.circs.Circ
        dataset: The dataset of Px values which to run the circuit with
        Nrange: A set of Nmax values to run the circuit at
        w: Baseline precision of the SC circuit
        max_var: Maximum variance for variance-based RET
    """

    _, Nmax, _ = N_from_trunc_err(ds, circ, e_min)
    _, Nmin, _ = N_from_trunc_err(ds, circ, e_max) #This only works for the RCED application....
    correct_vals = gen_correct(ds, circ)

    set_errs = []
    vret_errs = []
    pret_errs = []
    vret_ns = []
    pret_ns = []
    
    Nrange = [2 ** x for x in range(8)]
    for Nset in Nrange:
        err_set, err_vret, err_pret, N_vret, N_pret = \
        ET_sim_2(correct_vals, ds, circ, Nmin, Nset, Nmax, e_min, e_max)

        #errs
        set_errs.append(err_set)
        vret_errs.append(err_vret)
        pret_errs.append(err_pret)

        #Ns
        vret_ns.append(N_vret)
        pret_ns.append(N_pret)

        print("SC: err: {}, N: {}".format(err_set, Nmax))
        print("VRET: err: {}, N: {}".format(err_vret, N_vret))
        print("PRET: err: {}, N: {}".format(err_pret, N_pret))
    
    fig, ax = plt.subplots(1)
    plt.plot(Nrange, set_errs, label="SET", marker='o')
    plt.scatter(vret_ns[0], vret_errs[0], c='orangered', label="VRET", marker='+')
    plt.scatter(pret_ns, pret_errs, c='green', label="PRET", marker='*')
    #plt.axhline(y = max_var, color='purple', linestyle = '--', label="Max MSE")
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.legend()
    plt.xlabel("Bitstream length (N)")
    plt.ylabel("Error")
    plt.title("Error vs. Bitstream length for circuit: {}".format(circ.name))
    plt.show()

#LIB FUNC
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
    correct_vals = gen_correct(dataset, circ)
        
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

def fig_1(runs, Nmax = 64):
    #y axis: absolute error
    #x axis: bitstream length
    #show two traces: one px ~= 0.5 and one px ~= 0.9, demonstrating error converges differently
    #if possible, also show the error breakdown
    w = clog2(np.sqrt(Nmax))
    quant_err_1_array = []
    corr_err_array = []
    var_err_array = []
    total_err_array = []
    px = 0.1
    py = 0.5
    correct = px * py #replace with the correct function for the chosen 2-input circuit
    for N_et in range(1, Nmax):
        print(N_et)
        total_err = quant_err_1 = quant_err_2 = corr_err = var_err = 0.0
        for _ in range(runs):
            #px_trunc = fp_array(p_bin(px, w, lsb="right"))
            #py_trunc = fp_array(p_bin(py, w, lsb="right"))
            #trunc_correct = px_trunc * py_trunc

            bs_mat = lfsr_sng_efficient(np.array([px, py]), Nmax, clog2(Nmax), pack=False)
            bs_mat_et = bs_mat[:, :N_et]
            #full_output = np.mean(np.bitwise_and(bs_mat[0, :], bs_mat[1, :]))
            et_output = np.mean(np.bitwise_and(bs_mat_et[0, :], bs_mat_et[1, :]))
            
            total_err += np.abs(et_output - correct)
            #quant_err_1 += np.abs(full_output - correct)
            #quant_err_2 += np.abs(trunc_correct - correct)
            #corr_err += np.abs(et_output - np.mean(bs_mat_et[0, :]) * np.mean(bs_mat_et[1, :]))

            #hypergeo = (1/N_et) * (trunc_correct) * (1-trunc_correct) * ((Nmax - N_et)/(Nmax - 1))
            #var_err += np.sqrt(hypergeo)

        #quant_err_1_array.append(quant_err_1 / runs)
        #corr_err_array.append(corr_err / runs)
        #var_err_array.append(var_err / runs)
        total_err_array.append(total_err / runs) # if you need the total error

    total_err_array_2 = []
    px = 0.75
    py = 0.5
    correct_2 = px * py #replace with the correct function for the chosen 2-input circuit
    for N_et in range(1, Nmax):
        print(N_et)
        total_err = quant_err_1 = quant_err_2 = corr_err = var_err = 0.0
        for _ in range(runs):
            #px_trunc = fp_array(p_bin(px, w, lsb="right"))
            #py_trunc = fp_array(p_bin(py, w, lsb="right"))
            #trunc_correct = px_trunc * py_trunc

            bs_mat = lfsr_sng_efficient(np.array([px, py]), Nmax, clog2(Nmax), pack=False)
            bs_mat_et = bs_mat[:, :N_et]
            #full_output = np.mean(np.bitwise_and(bs_mat[0, :], bs_mat[1, :]))
            et_output = np.mean(np.bitwise_and(bs_mat_et[0, :], bs_mat_et[1, :]))
            
            total_err += np.abs(et_output - correct_2)
            #quant_err_1 += np.abs(full_output - correct)
            #quant_err_2 += np.abs(trunc_correct - correct)
            #corr_err += np.abs(et_output - np.mean(bs_mat_et[0, :]) * np.mean(bs_mat_et[1, :]))

            #hypergeo = (1/N_et) * (trunc_correct) * (1-trunc_correct) * ((Nmax - N_et)/(Nmax - 1))
            #var_err += np.sqrt(hypergeo)

        #quant_err_1_array.append(quant_err_1 / runs)
        #corr_err_array.append(corr_err / runs)
        #var_err_array.append(var_err / runs)
        total_err_array_2.append(total_err / runs) # if you need the total error

    

    # Labels and titles
    err_thresh = 0.05
    plt.axhline(y = err_thresh, color = 'r', label="Err thresh: {}".format(err_thresh), linestyle=(0, (3, 1, 1, 1)))

    N_et1 = 0
    for x, err in enumerate(total_err_array_2):
        if err < err_thresh:
            N_et1 = x
            break

    N_et2 = 0
    for x, err in enumerate(total_err_array):
        if err < err_thresh:
            N_et2 = x
            break

    plt.plot(total_err_array_2, color="darkblue", label=r"$Z_1$"+" where " + r"$P_{Z_1}=$"+"{}".format(np.round(correct_2, 2)))
    plt.plot(total_err_array, color="royalblue", label=r"$Z_2$"+" where " + r"$P_{Z_2}=$"+"{}".format(np.round(correct, 2)), ls=(0, (5, 1)))
    plt.axvline(x = N_et1, color = 'green', linestyle=(0, (1, 1)), label=r"$N_{ET1}$=" + "{}".format(N_et1))
    plt.axvline(x = N_et2, color = 'limegreen', linestyle=(0, (1, 1)), label=r"$N_{ET2}$="+ "{}".format(N_et2))
    plt.scatter([N_et1, N_et2], [err_thresh, err_thresh], c='red', s=30, zorder=5)
    plt.title("Error Analysis Over " + r"$N_{ET}$")
    plt.xlabel(r"$N_{ET}$")
    plt.ylabel('Error')
    #plt.legend(loc='upper right')

    # Print legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    print("\nLegend entries:")
    for label in labels:
        print(label)

    # Displaying the plot
    plt.show()

def scatter_ET_results(emin, emax):
    plt.axhline(y = emax, color = 'r', label=r"$\epsilon_{max}$ " + "{}".format(emax), linestyle=(0, (3, 1, 1, 1)))
    plt.axhline(y = emin, color = 'r', label=r"$\epsilon_{min}$ " + "{}".format(emin), linestyle=(0, (3, 1, 1, 1)))
    plt.title("")
    plt.legend()
    plt.show()

def error_bound_stats():
    #Testing done on 6/28/2024 relating to computing the error bound statistics

    ds = dataset_imagenet_samples(1, 1000, 2)
    circ = C_RCED()

    correct_vals = gen_correct(ds, circ)
    max_var = 0.01
    SC_vals, var_et_vals, var_et_Ns, cape_et_vals, cape_et_Ns, LD_et_vals, LD_et_Ns \
        = ET_sim(C_RCED(), ds, 32, 8, max_var=max_var)
    
    err_thresh = 0.01

    methods = [SC_vals, var_et_vals, cape_et_vals]
    ds_len = len(ds)
    for method in methods:
        perr = 0
        for i in range(ds_len):
            err = np.abs(method[i] - correct_vals[i])
            if err > err_thresh:
                perr += 1
        print(perr / ds_len)
    pass

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
    funcs = [lfsr, true_rand_hyper, counter, van_der_corput]
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