import numpy as np
from sim.Util import *
from sim.PCC import *
from sim.RNS import *
import experiments.early_termination.RET as RET
from synth.sat import sat, C_to_cgroups_and_sign

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
    """Stochastic Number Generator that produces correlated bitstreams.
    
    Args:
        rns: Random number source
        Cin: Correlation matrix (nv x nv) specifying input correlations.
             Values of 1 indicate positive correlation (same RNS source),
             values of -1 indicate negative correlation (inverted RNS bits),
             values of 0 indicate uncorrelated (different RNS sources).
        w: Bit width for the comparator
        nc: Number of constant (0.5-valued) inputs (default 0)
    """
    def __init__(self, rns: RNS, Cin, w, nc=0):
        self.rns = rns
        self.pcc = CMP_PCC(w)
        self.w = w
        self.nc = nc

        # Convert correlation matrix to cgroups and signs
        Cin = np.atleast_2d(Cin)
        self.nv = Cin.shape[0]
        self.n = self.nv + nc
        self.cgroups, self.signs = C_to_cgroups_and_sign(Cin)
        self.nv_star = len(np.unique(self.cgroups))
        self.rp_map = cgroups_to_rp_map(self.cgroups, w, nc)

        assert rns.full_width == self.rp_map.shape[0]
        assert self.nv * w + nc == self.rp_map.shape[1]

    def run(self, parr, N, **kwargs):
        if not isinstance(parr, list) and not isinstance(parr, np.ndarray): #handle the case where it's a single input
            parr = [parr]
        pbin = parr_bin(parr, self.w, lsb="left") #(nv x w)

        #Generate the random bits
        r = self.rns.run(N, **kwargs) #(wn* x N)
        r_mapped = r.T @ self.rp_map #(N x wnv+nc) = (N x wnv*+nc) (wnv*+nc x wnv+nc)
        #if w=3, n=2, rp is organized according to: [0 0 0|0 0 0]

        bs_mat = np.zeros((self.n, N), dtype=np.bool_)

        #variable inputs
        for i in range(self.nv):
            rbits = r_mapped[:, (i*self.w):(i*self.w+self.w)]
            if self.signs[i] == -1:
                rbits = np.bitwise_not(rbits)
            bs_mat[i, :] = self.pcc.run(rbits, pbin[i, :])

        #constant inputs
        for i in range(self.nc):
            bs_mat[i+self.nv, :] = r_mapped[:, self.w*self.nv+i]

        return bs_mat

def _get_rns_width(Cin, w, nc):
    """Helper to compute RNS width from correlation matrix."""
    Cin = np.atleast_2d(Cin)
    cgroups, _ = C_to_cgroups_and_sign(Cin)
    nv_star = len(np.unique(cgroups))
    return w * nv_star + nc

class HYPER_SNG(SNG):
    def __init__(self, w, Cin, nc=0, et=False, circ=None):
        self.et = et
        self._circ = circ  # Optional circuit for early termination
        super().__init__(HYPER_RNS(_get_rns_width(Cin, w, nc)), Cin, w, nc)

    def run(self, parr, N):
        if self.et:
            if self._circ is None:
                raise ValueError("Early termination requires a circ to be provided")
            Nret = RET.get_PRET_N(parr, self.w, self._circ)
            N = np.minimum(N, Nret)
        return super().run(parr, N)

class LFSR_SNG(SNG):
    def __init__(self, w, Cin, nc=0, et=False, circ=None):
        self.et = et
        self._circ = circ  # Optional circuit for early termination
        super().__init__(LFSR_RNS(_get_rns_width(Cin, w, nc)), Cin, w, nc)

    def run(self, parr, N, **kwargs):
        if self.et:
            if self._circ is None:
                raise ValueError("Early termination requires a circ to be provided")
            Nret = RET.get_PRET_N(parr, self.w, self._circ)
            N = np.minimum(N, Nret)
        return super().run(parr, N, **kwargs)

class RAND_SNG(SNG):
    """Generates zero autocorrelation."""
    def __init__(self, w, Cin, nc=0):
        super().__init__(RAND_RNS(_get_rns_width(Cin, w, nc)), Cin, w, nc)

    def run(self, parr, N):
        return super().run(parr, N)

class LFSR_SNG_N_BY_W(SNG):
    def __init__(self, w, Cin, nc=0):
        Cin = np.atleast_2d(Cin)
        cgroups, _ = C_to_cgroups_and_sign(Cin)
        nv_star = len(np.unique(cgroups))
        super().__init__(RNS_N_BY_W(LFSR_RNS, nv_star, nc, w), Cin, w, nc)

class COUNTER_SNG(SNG):
    """Generates maximum possible autocorrelation."""
    def __init__(self, w, Cin, nc=0):
        super().__init__(COUNTER_RNS(_get_rns_width(Cin, w, nc)), Cin, w, nc)

class VAN_DER_CORPUT_SNG(SNG):
    def __init__(self, w, Cin, nc=0):
        super().__init__(VAN_DER_CORPUT_RNS(_get_rns_width(Cin, w, nc)), Cin, w, nc)

class MIN_AUTOCORR_SNG(SNG):
    def __init__(self, w, Cin, nc=0):
        super().__init__(MIN_AUTOCORR_RNS(_get_rns_width(Cin, w, nc)), Cin, w, nc)

class PRET_SNG(SNG):
    def __init__(self, w, Cin, nc=0, et=True, lzd=False, lzd_func=None):
        """
        Args:
            w: Bit width
            Cin: Correlation matrix
            nc: Number of constant inputs
            et: Enable early termination
            lzd: Enable leading zero detection
            lzd_func: Function for LZD correction (required if lzd=True)
        """
        self.et = et
        self.lzd = lzd
        self._lzd_func = lzd_func
        super().__init__(BYPASS_COUNTER_RNS(_get_rns_width(Cin, w, nc)), Cin, w, nc)

    def run(self, parr, N):
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
            if self._lzd_func is None:
                raise ValueError("LZD requires lzd_func to be provided")
            lzd_bits = np.bitwise_not(np.bitwise_or.reduce(pbin, 0))
            found = False
            for wi in reversed(range(self.w)):
                if not lzd_bits[wi]:
                    found = True
                if found:
                    lzd_bits[wi] = False
            self.lzd_correction = self._lzd_func(2 ** np.sum(lzd_bits))
            lzd_bits = np.tile(lzd_bits, (self.nv_star))
            bp = np.bitwise_or(bp, lzd_bits)

        bp = np.concatenate((bp, np.zeros(self.nc, dtype=np.bool_)))
        self.rns.bp = bp

        if self.et:
            N = np.minimum(N, 2 ** np.sum(np.bitwise_not(self.rns.bp)))

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

def nonint_scc(bs_mat_uncorr, bs_mat_corr, c):
    """Given two sets of bitstreams that are uncorrelated and correlated respectively,
    generate a new pair of bitstreams consisting of a mixture of the two with a given non-integer SCC value
    
    This function is NOT meant to simulate the exact behavior of any stochastic circuit, 
    as it uses a built-in random function
    """
    n, N = bs_mat_uncorr.shape
    assert bs_mat_uncorr.shape == bs_mat_corr.shape

    bs_mat_out = np.zeros((n, N), dtype=np.bool_)
    for i in range(N):
        p = np.random.rand()
        if p < abs(c):
            bs_mat_out[:, i] = bs_mat_corr[:, i]
        else:
            bs_mat_out[:, i] = bs_mat_uncorr[:, i]
    return bs_mat_out

    """Code for testing this function:
        cin = np.array([
            [1, 0, -1],
            [0, 1, 0],
            [-1, 0, 1],
        ])
        cin2 = np.eye(3)
        sng = LFSR_SNG(8, cin)
        sng2 = LFSR_SNG(8, cin2)
        bs_mat = sng.run(np.array([0.75, 0.75, 0.33]), 4096)
        bs_mat2 = sng2.run(np.array([0.75, 0.75, 0.33]), 4096)
        print(scc_mat(bs_mat))
        print(scc_mat(bs_mat2))
        print(np.mean(bs_mat, axis=1))
        print(np.mean(bs_mat2, axis=1))
        bs_mat_out = nonint_scc(bs_mat, bs_mat2, 0.5)
        print(scc_mat(bs_mat_out))
        print(np.mean(bs_mat_out, axis=1))
        print(np.mean(bs_mat, axis=1))
    """