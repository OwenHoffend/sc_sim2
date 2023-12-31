import numpy as np
import matplotlib.pyplot as plt

from sim.RNS import *
from sim.deterministic import *
from sim.SCC import *
from sim.PCC import *
from sim.SNG import *
from sim.Util import *
from experiments.discrepancy import star_disc_2d, get_possible_Ps

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