import numpy as np

from sim.Util import p_bin
from sim.datasets import Dataset
from sim.SNG import SNG

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

def get_N_PRET(sng: SNG, ds: Dataset, trunc_w=None):
    """For a general dataset, computes the expected average bitstream length under PRET"""

    assert ds.n == sng.nv
    avg_N = 0.0
    for xs in ds:
        if trunc_w is not None:
            xs_trunc = list(map(lambda px: np.floor(px * 2 ** trunc_w) / (2 ** trunc_w), xs))
        else:
            xs_trunc = xs
        max_bits_used = np.zeros((sng.nv_star,))
        for i, x in enumerate(xs_trunc):
            bits_used = used_prec(x, sng.w)
            g = sng.cgroups[i]
            if bits_used > max_bits_used[g]:
                max_bits_used[g] = bits_used
        avg_N += 2 ** (np.sum(max_bits_used) + sng.nc)
    return avg_N / ds.num