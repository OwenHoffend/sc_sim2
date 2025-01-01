import numpy as np
import time
from sim.Util import *
from sim.PCC import *
from sim.RNS import *

#Tasks:
#TODO: These SNG library functions do a poor job of tiling for multiple-input circuits, should improve the codebase for this in the future

def sng_pack(bs_mat, pack, n):
    #Re-use some common code related to optional packing of SNG output bits
    if pack:
        return np.packbits(bs_mat, axis=1)
    else:
        if n == 1:
            return bs_mat.flatten()
        return bs_mat
    
def cgroups_to_rp_map(cgroups, w, nc):
    """cgroups specifies the correlation relationships between inputs as long as
        they consist of only SCC=0 or SCC=1. Each index corresponds to one of n outputs, and
        the number stored at that index corresponds to the index of the w-bit RNS source.
        For example, cgroups=[0, 1, 0, 1, 2] means inputs 0 and 2 are correlated, 1 and 3 are correlated,
        all other pairs are uncorrelated.

        --->> Constant inputs are NOT included in cgroups <<---

        rp_map is a Boolean matrix that specifies the individual wire connections between a 
        n_star*w bit rns and a n*w bit array of PCCs
    """
    nv_star = len(np.unique(cgroups))
    nv = len(cgroups)
    rp_map = np.zeros((w * nv_star + nc, w * nv + nc), dtype=np.bool_)
    for ni in range(nv):
        rns_src = cgroups[ni]
        for wi in range(w):
            rp_map[rns_src * w + wi, ni * w + wi] = True
    for nci in range(nc):
        rp_map[w * nv_star + nci, w * nv + nci] = True
    return rp_map

class SNG:
    def __init__(self, rns: RNS, pcc: PCC, nv, nc, w, cgroups):
        self.rns = rns
        self.pcc = pcc
        self.nv = nv
        self.nc = nc
        self.n = self.nv + self.nc
        self.w = w
        self.nc = nc

        #related to correlation
        self.cgroups = cgroups
        self.nv_star = len(np.unique(cgroups))
        self.rp_map = cgroups_to_rp_map(cgroups, w, nc)

        assert rns.full_width == self.rp_map.shape[0]
        assert nv * w + nc == self.rp_map.shape[1]

    def run(self, parr, N):
        pbin = parr_bin(parr, self.w, lsb="left") #(nv x w)

        #Generate the random bits
        r = self.rns.run(N) #(wn* x N)
        r_mapped = r.T @ self.rp_map #(N x wnv+nc) = (N x wnv*+nc) (wnv*+nc x wnv+nc)
        #if w=3, n=2, rp is organized according to: [0 0 0|0 0 0]

        bs_mat = np.zeros((self.n, N), dtype=np.bool_)

        #variable inputs
        for i in range(self.nv):
            bs_mat[i, :] = self.pcc.run(r_mapped[:, (i*self.w):(i*self.w+self.w)], pbin[i, :])

        #constant inputs
        for i in range(self.nc):
            bs_mat[i+self.nv, :] = r_mapped[:, self.w*self.nv+i]

        return bs_mat

class SNG_WN(SNG):
    def __init__(self, rns, w, circ):
        pcc = CMP_PCC(w)
        super().__init__(rns, pcc, circ.nv, circ.nc, w, circ.cgroups)  

class HYPER_SNG_WN(SNG_WN):
    def __init__(self, w, circ):
        super().__init__(HYPER_RNS_WN(circ.get_rns_width(w)), w, circ)

class LFSR_SNG_WN(SNG_WN):
    def __init__(self, w, circ):
        super().__init__(LFSR_RNS_WN(circ.get_rns_width(w)), w, circ)

class COUNTER_SNG_WN(SNG_WN):
    def __init__(self, w, circ):
        super().__init__(COUNTER_RNS_WN(circ.get_rns_width(w)), w, circ)

class VAN_DER_CORPUT_SNG_WN(SNG_WN):
    def __init__(self, w, circ):
        super().__init__(VAN_DER_CORPUT_RNS_WN(circ.get_rns_width(w)), w, circ)

class PRET_SNG_WN(SNG_WN):
    def __init__(self, w, circ, et=True, lzd=False):
        self.et = et
        self.lzd = lzd
        super().__init__(BYPASS_COUNTER_RNS_WN(circ.get_rns_width(w)), w, circ)

    def run(self, parr, Nmax):
        pbin = parr_bin(parr, self.w, lsb="left") #(nv x w)
        #0.375 @ 5 bits = 00110.0 --> 2 bits of TZD

        """Trailing zero detection"""
        bp = np.ones(self.rns.full_width - self.nc, dtype=np.bool_)
        for ni, g_idx in enumerate(self.cgroups):
            found_1 = False
            for wi in range(self.w):
                bp_idx = g_idx * self.w + wi
                b = pbin[ni, wi]
                if b:
                    found_1 = True
                bp[bp_idx] = bp[bp_idx] and not found_1

        """Leading zero detection"""
        if self.lzd:
            lzd_bits = np.bitwise_not(np.bitwise_or.reduce(pbin, 0))
            found = False
            for wi in reversed(range(self.w)):
                if not lzd_bits[wi]:
                    found = True
                if found:
                    lzd_bits[wi] = False
            lzd_bits = np.tile(lzd_bits, (self.nv_star))
            bp = np.bitwise_or(bp, lzd_bits)
            self.lzd_correction = 2 ** np.sum(lzd_bits)

        bp = np.concatenate((bp, np.zeros(self.nc, dtype=np.bool_)))
        self.rns.bp = bp

        if self.et:
            N = np.minimum(Nmax, 2 ** np.sum(np.bitwise_not(self.rns.bp)))
        else:
            N = Nmax
        return super().run(parr, N)
    
#Generate a bitstream with maximum possible streaming accuracy
def SA_sng(parr, N, w, pack=True):
    n = parr.size
    rsum = 0.0
    bs_mat = np.zeros((n, N), dtype=np.bool_)
    for i, px in enumerate(parr):
        for j in range(N):
            et_err0 = np.abs(rsum/(j+1.0)-px)
            et_err1 = np.abs((rsum+1)/(j+1.0)-px)
            if et_err1 < et_err0:
                rsum += 1.0
                bs_mat[i, j] = True
            else:
                bs_mat[i, j] = False

    return sng_pack(bs_mat, pack, n)