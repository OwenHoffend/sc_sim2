import numpy as np
import matplotlib.pyplot as plt

from sim.RNS import *
from sim.deterministic import *
from sim.Util import bit_vec_arr_to_int
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