from abc import abstractmethod
import numpy as np
from sim.circs.circs import mux, maj
from sim.Util import *

class PCC:
    def __init__(self, w):
        self.w = w

    @abstractmethod
    def run(self, r, p):
        pass

class CMP_PCC(PCC):
    def __init__(self, w):
        super().__init__(w)

    def run(self, r, p):
        #r: (N x w)
        #p: (1 x w)
        N = r.shape[0]
        r_ints = int_array(r)
        p_int = int_array(p)
        bs = np.zeros((N, ), dtype=np.bool_)
        for i in range(N):
            bs[i] = p_int > r_ints[i]
        return bs

def MMC(r, p, gamma):
    w = r.size
    z = False
    k = np.rint(w * gamma)
    for i in range(w):
        if i < w-k:
            z = mux(z, p[i], r[i])
        else:
            z = maj(z, p[i], r[i])
    return z

def CMP(r, p):
    return MMC(np.bitwise_not(r), p, 1)

def WBG(r, p):
    return MMC(r, p, 0)

def pcc(cs, val, precision):

    #Compute a pcc whose 0.5-constant values are given by cs, with any arbitrary value val
    #This is similar to the MMC function above, however it generates a chain of AND/OR gates based on the desired constant
    #Instead of generating the MUX/MAJ structure. So the circuit is specific to val, rather than being general

    if val == 0:
        return False
    elif val == 1:
        return True
    elif val == 0.5:
        return cs[0]

    radix_bits = p_bin(val, precision)
    while radix_bits[-1] == 0:
        radix_bits = radix_bits[:-1]
    actual_precision = radix_bits.size
    assert len(cs) >= actual_precision
    result = cs[0]

    radix_bits = radix_bits[:-1][::-1]
    for i in range(actual_precision-1):
        bit = radix_bits[i]
        if bit:
            result = np.bitwise_or(result, cs[i+1])
        else:
            result = np.bitwise_and(result, cs[i+1])
    return result