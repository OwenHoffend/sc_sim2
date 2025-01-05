import numpy as np

from sim.Util import p_bin
from sim.datasets import Dataset
from sim.circs.circs import *
from sim.sim import *

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

def analyze_PRET(max_w, circ: Circ, ds: Dataset, err_thresh):

    #Determine the correct w width to use based on SET
    #The error achieved by PRET is fixed by this choice
    PRET_w = max_w
    correct = gen_correct(circ, ds)
    ret_trunc = gen_correct(circ, ds, trunc_w=max_w)
    PRET_err = np.sqrt(MSE(ret_trunc, correct))
    for wi in reversed(range(max_w)):
        ret_trunc = gen_correct(circ, ds, trunc_w=wi)
        ret_trunc_err = np.sqrt(MSE(ret_trunc, correct))
        if ret_trunc_err < err_thresh:
            PRET_w = wi
            PRET_err = ret_trunc_err

    #Analyze TZD bit usage
    assert ds.n == circ.nv
    N_PRET = 0.0
    for xs in ds:
        xs_trunc = list(map(lambda px: np.floor(px * 2 ** PRET_w) / (2 ** PRET_w), xs))
        max_bits_used = np.zeros((circ.nv_star,))
        for i, x in enumerate(xs_trunc):
            bits_used = used_prec(x, max_w)
            g = circ.cgroups[i]
            if bits_used > max_bits_used[g]:
                max_bits_used[g] = bits_used
        N_PRET += 2 ** (np.sum(max_bits_used) + circ.nc)
    N_PRET /= ds.num

    #TODO: Analyze LZD bit usage

    return N_PRET, PRET_err, PRET_w