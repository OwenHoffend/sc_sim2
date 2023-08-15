import numpy as np
from pylfsr import LFSR
from sim.Util import bin_array

def lfsr(w, N):
    """
    w is the bit-width of the generator (this is a SINGLE RNS)
    N is the length of the sequence to sample (We could be sampling less than the full period of 2 ** w)
    """
    fpoly = LFSR().get_fpolyList(m=int(w))[0]
    all_zeros = np.zeros(w)
    while True:
        zero_state = np.random.randint(2, size=w) #Randomly decide where to put the init state
        if not np.all(zero_state == all_zeros):
            break
    while True:
        L = LFSR(fpoly=fpoly, initstate='random')
        if not np.all(L.state == all_zeros):
            break

    lfsr_bits = np.zeros((w, N), dtype=np.bool_)
    last_was_zero = False
    for i in range(N):
        if not last_was_zero and \
            np.all(L.state == zero_state):
                lfsr_bits[:, i] = all_zeros
                last_was_zero = True
                continue
        last_was_zero = False
        L.runKCycle(1)
        lfsr_bits[:, i] = L.state
    return lfsr_bits

def true_rand(w, N):
    assert N <= 2 ** w
    nums = np.array([x for x in range(2 ** w)])
    np.random.shuffle(nums)
    nums = nums[:N] #only the first N entries
    rns_bits = np.empty((w, N), dtype=np.bool_)
    for i in range(N):
        rns_bits[:, i] = bin_array(nums[i], w)
    return rns_bits

def counter(w, N):
    assert N <= 2 ** w
    rns_bits = np.empty((w, N), dtype=np.bool_)
    for i in range(N):
        rns_bits[:, i] = bin_array(i, w)
    return rns_bits

def van_der_corput(w, N):
    assert N <= 2 ** w
    rns_bits = np.empty((w, N), dtype=np.bool_)
    for i in range(N):
        rns_bits[:, i] = bin_array(i, w)[::-1]
    return rns_bits