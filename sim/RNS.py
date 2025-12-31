from abc import abstractmethod
import numpy as np
from pylfsr import LFSR
from sim.Util import bin_array, int_array
from typing import Type
from sim.circs.circs import *

class RNS:
    def __init__(self, full_width):
        self.full_width = full_width
        self.full_period = 2 ** self.full_width

    @abstractmethod
    def run(self, N):
        pass

class RNS_N_BY_W(RNS):
    def __init__(self, rns: Type[RNS], circ: Circ, w):
        super().__init__(circ.get_rns_width(w))
        self.rns = rns
        self.nv_star = circ.nv_star
        self.nc = circ.nc
        self.w = w

    def run(self, N):
        """Use nv RNSes of width w plus one of width min(nc, 4) for constant"""
        #rns_bits = np.zeros((self.nv * self.w, N))
        #for i in range(self.nv):
        #    rns_bits[i * self.w:(i+1)*self.w, :] = self.rns(self.w).run(N)
        #
        #const_w = np.maximum(4, self.nc)
        #const_bits = self.rns(const_w).run(N)
        #rns_bits = np.concatenate((rns_bits, const_bits[:self.nc, :]))


        """Use nv-1 RNSes of width w plus one of width w+nc"""
        if self.nv_star < 1:
            raise NotImplementedError("RNS_N_BY_W not implemented for circuits with only constant inputs")

        rns_bits = np.zeros(((self.nv_star - 1) * self.w, N))
        for i in range(self.nv_star - 1):
            rns_bits[i * self.w:(i+1)*self.w, :] = self.rns(self.w).run(N)

        const_w = self.w + self.nc
        const_bits = self.rns(const_w).run(N)
        rns_bits = np.concatenate((rns_bits, const_bits))

        return rns_bits
    
class RNS_CLOCK_ROTATE(RNS):
    def __init__(self, rns: Type[RNS], circ: Circ, w):
        super().__init__(circ.get_rns_width(w))
        self.rns = rns
        self.nv_star = circ.nv_star
        self.nc = circ.nc
        self.w = w

    def run(self, N):
        pass

class LFSR_RNS(RNS):
    """One single wnv*+nc-bit LFSR shared among all the inputs"""

    def __init__(self, full_width):
        super().__init__(full_width)
        self.fpoly_cache = {}

    def run(self, N, poly_idx=1, use_rand_init=True, add_zero_state=True):
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
        if use_rand_init:
            while True:
                zero_state = np.random.randint(2, size=self.full_width) #Randomly decide where to put the zero state
                if not np.all(zero_state == all_zeros):
                    break
            while True:
                init_state = np.random.randint(2, size=self.full_width) #Randomly pick an init state
                if not np.all(init_state == all_zeros):
                    break
        else:
            zero_state = all_zeros
            zero_state[0] = 1
            init_state = all_zeros
            init_state[0] = 1

        L = LFSR(fpoly=fpoly, initstate=init_state)

        lfsr_bits = np.zeros((self.full_width, N), dtype=np.bool_)
        last_was_zero = False
        for i in range(N):
            if add_zero_state and not last_was_zero and \
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

class HYPER_RNS(RNS):
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
    
class RAND_RNS(RNS):
    def run(self, N):
        rns_bits = np.empty((self.full_width, N), dtype=np.bool_)
        for i in range(N):
            rns_bits[:, i] = np.random.randint(2, size=self.full_width)
        return rns_bits
    
class COUNTER_RNS(RNS):
    """One single wnv*+nc-bit counter, repeats if N > full period"""

    def run(self, N):
        rns_bits = np.empty((self.full_width, N), dtype=np.bool_)
        for i in range(N):
            rns_bits[:, i] = bin_array(i % self.full_period, self.full_width)
        return rns_bits
    
class VAN_DER_CORPUT_RNS(RNS):
    def run(self, N):
        rns_bits = np.empty((self.full_width, N), dtype=np.bool_)
        for i in range(N):
            rns_bits[:, i] = bin_array(i % self.full_period, self.full_width)[::-1]
        return rns_bits

class MIN_AUTOCORR_RNS(RNS):
    def run(self, N):
        cnt_bits = COUNTER_RNS(self.full_width).run(N)
        up = cnt_bits[:, ::2]
        down = cnt_bits[:, 1::2][:, ::-1]
        rns_bits = np.empty((self.full_width, N), dtype=np.bool_)
        rns_bits[:, ::2] = up
        rns_bits[:, 1::2] = down
        return rns_bits
    
class BYPASS_COUNTER_RNS(RNS):
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