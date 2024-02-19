import numpy as np
from sim.Util import clog2, parr_bin, bin_array
from sim.PCC import CMP

def check_ATPP(bs_mat, Bx):
    n, N = bs_mat.shape
    log2_N = clog2(N)
    correct = np.zeros((n, ))
    for i in range(1, log2_N):
        correct += (2 ** (-i)) * Bx[:, i-1]
        prec = np.mean(bs_mat[:, :(2**i)], axis=1)
        print(prec)
        assert np.all(prec == correct)

def print_precision_points(bs_mat):
    n, N = bs_mat.shape
    log2_N = clog2(N)
    for i in range(1, log2_N+1):
        print(np.mean(bs_mat[:, :(2**i)], axis=1))

#2/18/2024 implementation of the CAPE-based early-terminating SNG
ctr_cache = {}
def CAPE_ET(parr, w_, cgroups, Nmax=None, et=False, use_wbg=False, return_N_only=False):
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
        wmax = np.ceil(clog2(Nmax) / s).astype(np.int32) #maximum precision used for Nmax
        w = np.minimum(w_, wmax)
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
        for n_i, g in enumerate(cgroups):
            if last_g is not None and g != last_g: #new uncorrelated group
                s_i += 1
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
    else:
        w_actual = ctr_width
        N = Nmax

    if return_N_only:
        return N

    global ctr_cache
    if w_actual in ctr_cache:
        ctr_list = ctr_cache[w_actual]
    else:
        ctr_list = [bin_array(i, w_actual, lsb='left') for i in range(N)]
        ctr_cache[w_actual] = ctr_list

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

    return bs_mat
