import numpy as np
from pylfsr import LFSR
from sim.Util import bin_array, int_array

fpoly_cache = {}
def lfsr(w, N, poly_idx=0, use_rand_init=True):
    """
    w is the bit-width of the generator (this is a SINGLE RNS)
    N is the length of the sequence to sample (We could be sampling less than the full period of 2 ** w)
    """
    cache_str = str(w) + ":" + str(poly_idx)
    if cache_str in fpoly_cache: #this optimization greatly speeds up the lfsr code :)
        fpoly = fpoly_cache[cache_str]
    else:
        fpoly = LFSR().get_fpolyList(m=int(w))[poly_idx]
        fpoly_cache[cache_str] = fpoly
        
    all_zeros = np.zeros(w)
    while True:
        zero_state = np.random.randint(2, size=w) #Randomly decide where to put the zero state
        if not np.all(zero_state == all_zeros):
            break

    if use_rand_init:
        while True:
            init_state = np.random.randint(2, size=w) #Randomly pick an init state
            if not np.all(init_state == all_zeros):
                break
    else:
        init_state = np.zeros((w,))
        init_state[0] = 1

    L = LFSR(fpoly=fpoly, initstate=init_state)

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

def print_all_fpolys_hex():
    polys = LFSR().get_fpolyList()
    for w, poly_list in polys.items():
        print("localparam [{}:0] LFSR_{}_POLYS[{}] = '{{".format(w-1, w, len(poly_list)))
        for idx, poly in enumerate(poly_list):
            #print(poly)
            p = ["0" for _ in range(w)]
            for i in poly:
                if i != w:
                    p[(w-1)-i] = "1"
            p[w-1] = "1"
            p = ''.join(p)
            #print(p)
            padding = '0' * ((4 - len(p) % 4) % 4)  # Compute necessary padding
            padded_bit_string = padding + p

            # Now convert the padded bit string
            bit_int = int(padded_bit_string, 2)
            if idx == len(poly_list) - 1:
                print("\t{}'h{}".format(w, format(bit_int, 'x')))
            else:
                print("\t{}'h{},".format(w, format(bit_int, 'x')))
        print("};")

def is_complete_sequence(bmat):
    """Test to see if a bmat contains all possible states of w bits"""
    w, N = bmat.shape
    imat = int_array(bmat.T)
    unq = np.unique(imat)
    return np.all(unq == np.array([x for x in range(2 ** w)]))

def true_rand_hyper(w, N):
    #assert N <= 2 ** w
    nums = np.array([x for x in range(2 ** w)] * np.rint(N / w).astype(np.int32))
    np.random.shuffle(nums)
    #nums = nums[:N] #only the first N entries
    rns_bits = np.empty((w, N), dtype=np.bool_)
    for i in range(N):
        rns_bits[:, i] = bin_array(nums[i], w)
    return rns_bits

def true_rand(w, N):
    rns_bits = np.empty((w, N), dtype=np.bool_)
    for i in range(w):
        rns_bits[i, :] = np.random.choice([False, True], size=N)
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