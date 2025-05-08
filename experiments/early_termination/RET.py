import numpy as np

from sim.Util import p_bin
from sim.datasets import Dataset
from sim.circs.circs import Circ
import sim.sim as sim

def used_prec(x, w, lzd=False):
    bin_rep = p_bin(x, w, lsb="right")
    bits_used = 0
    for i in reversed(range(w)):
        if bin_rep[i]:
            bits_used = i+1
            break
    else:
        return 0

    if lzd:
        lzd_bits = 0
        for i in range(w):
            if bin_rep[i]:
                break
            lzd_bits += 1
        bits_used -= lzd_bits

    return bits_used

def get_PRET_N(xs, w, circ: Circ):
    max_bits_used = np.zeros((circ.nv_star,))
    for i, x in enumerate(xs):
        bits_used = used_prec(x, w)
        g = circ.cgroups[i]
        if bits_used > max_bits_used[g]:
            max_bits_used[g] = bits_used
    return int(2 ** (np.sum(max_bits_used) + circ.nc))

def get_PRET_w(max_w, circ: Circ, ds: Dataset, err_thresh, return_all_errs=False, use_cache=False):

    #Determine the correct w width to use based on SET
    #The error achieved by PRET is fixed by this choice
    PRET_w = max_w
    correct = sim.gen_correct(circ, ds, use_cache=use_cache)
    ret_trunc = sim.gen_correct(circ, ds, trunc_w=max_w, use_cache=use_cache)
    PRET_err = np.sqrt(sim.MSE(ret_trunc, correct))

    all_errs = []
    for wi in reversed(range(max_w + 1)):
        ret_trunc = sim.gen_correct(circ, ds, trunc_w=wi)
        ret_trunc_err = np.sqrt(sim.MSE(ret_trunc, correct))
        #ret_trunc_err = np.max(np.abs(ret_trunc - correct))
        if ret_trunc_err < err_thresh:
            PRET_w = wi
            PRET_err = ret_trunc_err
        all_errs.append(ret_trunc_err)

    if return_all_errs:
        PRET_err = all_errs
    return PRET_err, PRET_w

def analyze_PRET(max_w, circ: Circ, ds: Dataset, err_thresh, **kwargs):

    PRET_err, PRET_w = get_PRET_w(max_w, circ, ds, err_thresh, kwargs)

    if "Nset" in kwargs:
        Nset = kwargs["Nset"]
    else:
        Nset = np.inf

    #Analyze TZD bit usage
    #assert ds.n == circ.nv
    N_PRET = 0.0
    for xs in ds:
        xs = circ.parr_mod(xs)
        N_PRET += np.minimum(get_PRET_N(xs, PRET_w, circ), Nset)
        #xs_trunc = list(map(lambda px: np.floor(px * 2 ** PRET_w) / (2 ** PRET_w), xs))
        #max_bits_used = np.zeros((circ.nv_star,))
        #for i, x in enumerate(xs_trunc):
        #    bits_used = used_prec(x, max_w)
        #    g = circ.cgroups[i]
        #    if bits_used > max_bits_used[g]:
        #        max_bits_used[g] = bits_used
        #N_PRET += np.minimum(2 ** (np.sum(max_bits_used) + circ.nc), Nset)
    N_PRET /= ds.num

    return N_PRET, PRET_err, PRET_w