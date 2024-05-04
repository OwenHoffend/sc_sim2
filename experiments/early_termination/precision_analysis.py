import numpy as np
from sim.Util import bin_array
from experiments.early_termination.et_hardware import *
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mc

def bseq(w):
    """Returns the sequence of the number of binary bits used for all possible values of a binary string of width w"""
    if w == 0:
        return np.array([0,])
    if w == 1:
        return np.array([0, 1])
    else:
        bseq_prev = bseq(w-1)
        new_entries = np.full((2 ** (w-1), ), w)
        bseq_new = np.empty((2 ** w,), dtype=bseq_prev.dtype)
        bseq_new[0::2] = bseq_prev
        bseq_new[1::2] = new_entries
        return bseq_new
    
def bseq_multi(w, n):
    base = bseq(w)
    if n == 1:
        return base
    res = np.add.outer(base, base)
    for _ in range(n-2):
        res = np.add.outer(base, res)
    return res

def bseq_2_corr(w):
    base = bseq(w)
    res = np.empty((2 ** w, 2 ** w))
    for i in range(2 ** w):
        for j in range(2 ** w):
            res[i, j] = np.maximum(base[i], base[j])
    return res
    
def mbseq(w, n):
    return np.mean(2 ** bseq_multi(w, n) / 2 ** (w*n))

def used_prec(x, w):
    bin_rep = p_bin(x, w, lsb="right")
    bits_used = 0
    for i in reversed(range(w)):
        if bin_rep[i]:
            bits_used = i+1
            break
    else:
        bits_used = 0
    return bits_used

def avg_used_prec(xs, w):
    #For a general dataset, gets the average number of actual bits used
    #after truncating the precision to w

    avg_bits_used = 0
    for x in xs:
        bits_used = used_prec(x, w)
        avg_bits_used += bits_used
    avg_bits_used /= len(xs)
    print(avg_bits_used)
    return avg_bits_used

def ret_vs_set_1d(w):
    #prints the average error obtained when early terminating based on the average *actual* required precision
    #This assumes a uniform input weighting

    xs = [x/(2**w) for x in range(2 ** w)]
    req_precs = bseq(w)
    avg_prec = np.rint(np.mean(req_precs)).astype(np.int32)

    avg_err = 0.0
    for x in xs:
        bin_rep = p_bin(x, avg_prec, lsb="right")
        restored = fp_array(bin_rep)
        avg_err += np.abs(restored - x)
    avg_err /= 2 ** w
    print(avg_err)

def heatmap(xs, ys, zs, zmin=None, zmax=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cmap = "Blues" #ERRs
    y, x = np.meshgrid(xs, ys)

    img = ax.imshow(zs, cmap=cmap, origin="lower", extent=[0, 1, 0, 1])

    # Remove ticks by setting their length to 0
    #ax.yaxis.set_tick_params(length=0)
    #ax.xaxis.set_tick_params(length=0)

    cb = plt.colorbar(img, ax=[ax],location='left', label="Number of stochastic bits", pad=0.12)

    plt.xlabel("Px value")
    plt.ylabel("Py value")
    plt.title("Required stochastic bits vs. Px/Py values, SCC=1")
    plt.show()

def heatmap_actual_precision_use(w):
    xs = [x/(2**w) for x in range(2 ** w)]

    #req_precs = bseq_multi(w, 2)
    req_precs = bseq_2_corr(w)
    zs = 2 ** req_precs

    print(np.mean(req_precs)) #on average, 8 bits out of the total 10 bits
    print(req_precs)

    heatmap(xs, xs, zs)

