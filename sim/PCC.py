import numpy as np
from sim.circs import mux, maj

def MMC(r, p, gamma):
    w = r.size
    z = False
    k = np.rint(w * gamma)
    for i in range(w):
        if i < w-k:
            z = mux(r[i], z, p[i])
        else:
            z = maj(r[i], z, p[i])
    return z

def CMP(r, p):
    return MMC(r, p, 1)

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

    radix_bits = np.zeros(precision, dtype=np.bool_)
    cmp = 0.5
    for i in range(precision):
        if val >= cmp:
            radix_bits[i] = 1
            val -= cmp
        else:
            radix_bits[i] = 0
        cmp /= 2
    while radix_bits[-1] == 0:
        radix_bits = radix_bits[:-1]
    precision = radix_bits.size
    actual_precision = precision
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