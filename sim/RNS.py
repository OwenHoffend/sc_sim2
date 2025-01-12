from abc import abstractmethod
import numpy as np
from pylfsr import LFSR
from sim.Util import bin_array, int_array

class RNS:
    def __init__(self, full_width):
        self.full_width = full_width
        self.full_period = 2 ** self.full_width

    @abstractmethod
    def run(self, N):
        pass

class LFSR_RNS_WN(RNS):
    """One single wnv*+nc-bit LFSR shared among all the inputs"""

    def __init__(self, full_width):
        super().__init__(full_width)
        self.fpoly_cache = {}

    def run(self, N, poly_idx=0, use_rand_init=True):
        """
        w is the bit-width of the generator (this is a SINGLE RNS)
        N is the length of the sequence to sample (We could be sampling less than the full period of 2 ** w)
        """

        if self.full_width > 32:
            raise ValueError("LFSR width of {} is not supported".format(self.full_width))

        cache_str = str(self.full_width) + ":" + str(poly_idx)
        if cache_str in self.fpoly_cache: #this optimization greatly speeds up the lfsr code :)
            fpoly = self.fpoly_cache[cache_str]
        else:
            fpoly = LFSR().get_fpolyList(m=int(self.full_width))[poly_idx]
            self.fpoly_cache[cache_str] = fpoly
            
        all_zeros = np.zeros(self.full_width)
        while True:
            zero_state = np.random.randint(2, size=self.full_width) #Randomly decide where to put the zero state
            if not np.all(zero_state == all_zeros):
                break

        if use_rand_init:
            while True:
                init_state = np.random.randint(2, size=self.full_width) #Randomly pick an init state
                if not np.all(init_state == all_zeros):
                    break
        else:
            init_state = np.zeros((self.full_width,))
            init_state[0] = 1

        L = LFSR(fpoly=fpoly, initstate=init_state)

        lfsr_bits = np.zeros((self.full_width, N), dtype=np.bool_)
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
    """Helper function for generating verilog LFSR polynomials"""

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

class HYPER_RNS_WN(RNS):
    """One single wnv*+nc-bit ideal hypergeometric source shared among all the inputs
        Repeats if N > full period
    """

    def run(self, N):
        nums = np.array([x for x in range(self.full_period)])
        np.random.shuffle(nums)
        rns_bits = np.empty((self.full_width, N), dtype=np.bool_)
        for i in range(N):
            rns_bits[:, i] = bin_array(nums[i % self.full_period], self.full_width)
        return rns_bits
    
class COUNTER_RNS_WN(RNS):
    """One single wnv*+nc-bit counter, repeats if N > full period"""

    def run(self, N):
        rns_bits = np.empty((self.full_width, N), dtype=np.bool_)
        for i in range(N):
            rns_bits[:, i] = bin_array(i % self.full_period, self.full_width)
        return rns_bits
    
class VAN_DER_CORPUT_RNS_WN(RNS):
    def run(self, N):
        rns_bits = np.empty((self.full_width, N), dtype=np.bool_)
        for i in range(N):
            rns_bits[:, i] = bin_array(i % self.full_period, self.full_width)[::-1]
        return rns_bits
    
class BYPASS_COUNTER_RNS_WN(RNS):
    def __init__(self, full_width):
        self.bp = np.zeros(full_width, dtype=np.bool_) #should be overridden to function properly
        super().__init__(full_width)

    def run(self, N):
        equivalent_width = np.sum(~self.bp)
        equivalent_period = 2 ** equivalent_width
        cnt_bits = np.empty((equivalent_width, N), dtype=np.bool_)
        for i in range(N):
            cnt_bits[:, i] = bin_array(i % equivalent_period, equivalent_width)

        rns_bits = np.zeros((self.full_width, N))
        j = 0
        for i in range(self.full_width):
            if ~self.bp[i]:
                rns_bits[i, :] = cnt_bits[j, :]
                j += 1
        return rns_bits

def is_complete_sequence(bmat):
    """Test to see if a bmat contains all possible states of w bits"""
    w, N = bmat.shape
    imat = int_array(bmat.T)
    unq = np.unique(imat)
    return np.all(unq == np.array([x for x in range(2 ** w)]))