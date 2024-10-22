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

def sng(parr, N, w, rns, pcc, corr=0, cgroups=None, pack=True):
    n = parr.size
    pbin = parr_bin(parr, w, lsb="left")
    
    #Generate the random bits
    bs_mat = np.zeros((n, N), dtype=np.bool_)
    r = rns(w, N)

    if cgroups is not None:
        g = cgroups[0]
    for i in range(n):
        if cgroups is not None:
            if cgroups[i] != g:
                r = rns(w, N)
                g = cgroups[i]
        elif not corr: #if not correlated, get a new independent rns sequence
            r = rns(w, N)

        p = pbin[i, :]
        for j in range(N):
            bs_mat[i, j] = pcc(r[:, j], p)

    return sng_pack(bs_mat, pack, n)

#def sng_precise_sample(parr, rns, N, w, circ, pack=False):
#    pbin = parr_bin(parr, w, lsb="left")
#    
#    #Generate the random bits
#    bs_mat = np.zeros((circ.n, N), dtype=np.bool_)
#    r = rns(circ.get_rns_width(w), N)
#    for i in range(circ.n):
#        p = pbin[i, :]
#        for j in range(N):
#            bs_mat[i, j] = pcc(r[i:i+w, j], p)
#
#    return sng_pack(bs_mat, pack, n)
    
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

def true_rand_sng_efficient(parr, N, w, corr=0, cgroups=None, pack=True):
    n = parr.size
    pbin = parr_bin(parr, w, lsb="left")
    pbin_ints = int_array(pbin)
    
    #Generate the random bits
    bs_mat = np.zeros((n, N), dtype=np.bool_)
    r = true_rand_hyper(w, N)
    r_ints = int_array(r.T)

    if cgroups is not None:
        g = cgroups[0]
    for i in range(n):
        if cgroups is not None:
            if cgroups[i] != g:
                g = cgroups[i]
                r = true_rand_hyper(w, N)
                r_ints = int_array(r.T)
        elif not corr: #if not correlated, get a new independent rns sequence
            r = true_rand_hyper(w, N)
            r_ints = int_array(r.T)

        p = pbin_ints[i]
        for j in range(N):
            bs_mat[i, j] = p > r_ints[j]

    return sng_pack(bs_mat, pack, n)

def lfsr_sng(parr, N, w, **kwargs):
    return sng(parr, N, w, lfsr, CMP, **kwargs)

def van_der_corput_sng(parr, N, w, **kwargs):
    return sng(parr, N, w, van_der_corput, CMP, **kwargs)

def counter_sng(parr, N, w, **kwargs):
    return sng(parr, N, w, counter, CMP, **kwargs)

def true_rand_hyper_sng(parr, N, w, **kwargs):
    return sng(parr, N, w, true_rand_hyper, CMP, **kwargs)

def true_rand_sng(parr, N, w, **kwargs):

    #RNS source is truly 0.5 random
    return sng(parr, N, w, true_rand, CMP, **kwargs)

def binomial_sng(parr, N):
    #w doesn't matter for this one, as we are just generating the sequence for each input
    n = parr.size
    bs_mat = np.zeros((n, N), dtype=np.bool_)
    for i in range(n):
        bs_mat[i, :] = np.random.choice([False, True], size=N, p=[1-parr[i], parr[i]])
    return bs_mat

#2/18/2024 implementation of the CAPE-based early-terminating SNG
ctr_cache = {}
def CAPE_sng(parr, w_, cgroups, Nmax=None, pack=False, et=False, use_wbg=False, return_N_only=False, use_consensus_for_corr=False):
    """cgroups defines the correlation structure. It should have the same length as n
        entries with the same value of cgroups correspond to correlated inputs:

        Example: [0,0,0,1,2,3] indicates 6 inputs: the first 3 are correlated and the last are uncorrelated
        for 4 total uncorrelated groups (s=4)
    """

    """Step 1: Compute the input fixed-point binary matrix
        Truncate the matrix according to a maximum bitstream length Nmax
        This truncation corresponds to a static early termination operation
    """
    s = np.unique(cgroups).size
    if Nmax is not None: #optional parameter to specify a maximum bitstream length
        #wmax = np.ceil(clog2(Nmax) / s).astype(np.int32) #maximum precision used for Nmax
        #w = np.minimum(w_, wmax)
        w = (clog2(Nmax) / s).astype(np.int32)
        ctr_width = s * w
    else:
        w = w_
        Nmax = 2 ** (s * w)
        ctr_width = s * w
    Bx = parr_bin(parr, w, lsb="right")
    
    """Step 2: Trailing zero detection: 
        First, evaluate the amount of precision actually required by performing trailing zero
        detection on Bx with a right-hand MSB.
        Example: [False, True, False, False] --> [False, False, True, True]

        Groups that are correlated are first ORed together, as the required precision is set by the
        input within the group that uses the highest precision
    """
    if et:
        Bx_groups = np.zeros((s, w), dtype=np.bool_)
        last_g = None
        s_i = 0
        if use_consensus_for_corr:
            consensus = np.zeros((w, ), dtype=np.int32)
            consensus_sz = 0
        for n_i, g in enumerate(cgroups):
            if last_g is not None and g != last_g: #new uncorrelated group
                s_i += 1
                if use_consensus_for_corr:
                    consensus_sz = 0

            if use_consensus_for_corr:
                consensus_sz += 1
                consensus += np.bitwise_not(Bx[n_i, :])
                consensus_thresh = np.floor(consensus_sz / 2).astype(np.int32) + 1
                Bx_groups[s_i, :] = consensus < consensus_thresh
            else:
                Bx_groups[s_i, :] = np.bitwise_or(Bx_groups[s_i, :], Bx[n_i, :])
            last_g = g

        tzd = np.zeros((s, w), dtype=np.bool_)
        col = np.zeros((s, ), dtype=np.bool_)
        for i in reversed(range(w)):
            col = np.bitwise_or(Bx_groups[: , i], col)
            tzd[:, i] = np.bitwise_not(col)
        
        """Step 3: Generate the counter sequence with bits bypassed due to the tzd from step 2
            This works by first generating a counter sequence of a width equal to the precision
            that's actually required, then padding the result with zeros in the correct locations
        """
        bp = tzd.reshape((ctr_width), order='F')
        w_actual = ctr_width - np.sum(bp)
        N = 2 ** np.minimum(clog2(Nmax), w_actual)
        #print("CAPE ET at : {} out of {}".format(N, Nmax))
    else:
        w_actual = ctr_width
        N = Nmax

    if return_N_only: #TODO: should match the analytical approach, test this
        return N

    global ctr_cache
    cache_str = '{}_{}'.format(w_actual, N)
    if cache_str in ctr_cache:
        ctr_list = ctr_cache[cache_str]
    else:
        ctr_list = [bin_array(i, w_actual, lsb='left') for i in range(N)]
        ctr_cache[cache_str] = ctr_list

    ctr = np.array(ctr_list)
    if not use_wbg:
        ctr = np.flip(ctr, axis=0)

    if et:
        bypassed_ctr = np.zeros((N, ctr_width), dtype=np.bool_)
        j = 0
        for i in range(ctr_width):
            if bp[i]:
                bypassed_ctr[:, i] = np.zeros((N), dtype=np.bool_)
            else:
                bypassed_ctr[:, i] = ctr[:, j]
                j += 1
        ctr = bypassed_ctr

    """Step 4: Evaluate the CMP/WBG operation for the counter bits, 
        Use RNS sharing where necessary to induce correlation
    """
    n = parr.size
    bs_mat = np.zeros((n, N), dtype=np.bool_)

    for i in range(N):
        last_g = None
        s_j = 0
        ci = ctr[i, :]
        for n_j, g in enumerate(cgroups):
            if g != last_g: #new uncorrelated group
                r = ci[s_j::s] #strided access (width of w)
                s_j += 1
            p = Bx[n_j, :] #width of w
            last_g = g
            
            if use_wbg: #WBG mode (sacrifices some SCC=1 for better area)
                wbg = False
                nands = True
                for k in range(w):
                    wbg = wbg or (r[k] and p[k] and nands)
                    nands = nands and not r[k]
                bs_mat[n_j, i] = wbg
            else: #CMP mode
                cmp = False
                for k in reversed(range(w)):
                    if r[k] and not p[k]:
                        cmp = False
                    elif p[k] and not r[k]:
                        cmp = True
                bs_mat[n_j, i] = cmp
    
    return sng_pack(bs_mat, pack, n)
    
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

def sng_from_pointcloud(parr, S, pack=True):
    _, N = S.shape
    n = parr.size
    bs_mat = np.zeros((n, N), dtype=np.bool_)
    parr *= (N ** (1/n))
    for i in range(N):
        bs_mat[:, i] = S[:, i] < parr 

    return sng_pack(bs_mat, pack, n)