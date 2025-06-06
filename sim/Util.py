import numpy as np
import matplotlib.pyplot as plt

def clog2(N):
    return np.ceil(np.log2(N)).astype(np.int32)

def bin_array(num, m, lsb='left'):
    """Convert a positive integer num into an m-bit bit vector"""
    arr = np.array(
        list(np.binary_repr(num).zfill(m))
    ).astype(bool)

    if lsb == 'left':
        return arr[::-1]
    else:
        return arr

def int_array(bmat, lsb='left'):
    "Convert a bin_array back to an int one"
    if len(bmat.shape) == 1:
        n = bmat.size
    else:
        _, n = bmat.shape
    bmap = np.array([1 << x for x in range(n)])
    if lsb == 'right':
        bmap = np.flip(bmap)
    return (bmat @ bmap).astype(int)

def fp_array(bmat):
    "same thing as int_array, but interpret as a fixed-point binary fraction"
    if len(bmat.shape) == 1:
        n = bmat.size
    else:
        _, n = bmat.shape
    bmap = np.array([1.0 / (2.0 ** (x+1)) for x in range(n)])
    return bmat @ bmap

def p_bin(p, w, lsb="left"):
    "1D case of parr_bin"
    return parr_bin(np.array([p, ]), w, lsb)[0]

def parr_bin(parr, w, lsb="left"):
    "convert an array of probability values to fixed-point binary at the specified precision"

    n = len(parr)
    parr_bin = np.zeros((n, w), dtype=np.bool_)
    for i in range(n):
        cmp = 0.5
        p = parr[i]

        if lsb == "left":
            r = reversed(range(w))
        else: 
            r = range(w)

        for j in r:
            if p >= cmp:
                parr_bin[i, j] = True
                p -= cmp
            else:
                parr_bin[i, j] = False
            cmp /= 2
    return parr_bin

bv_int_cache = {}
def bit_vec_to_int(vec, lsb='left'):
    """Utility function for converting a np array bit vector to an integer"""
    str_vec = "".join([str(x) for x in vec])
    if str_vec in bv_int_cache.keys():
        return bv_int_cache[str_vec]
    if lsb != 'left':
        vec = vec[::-1]
    result = vec.dot(2**np.arange(vec.size))
    bv_int_cache[str_vec] = result 
    return result

def bit_vec_arr_to_int(arr, lsb='left'):
    _, N = arr.shape
    result = np.zeros(N, dtype=np.int32)
    for i in range(N):
        result[i] = bit_vec_to_int(arr[:, i], lsb=lsb)
    return result

def MSE(a, b):
    return np.mean((a - b) ** 2)

def bs_delay(bs, n):
    #input bs 1010 output bs 0000...1010
    return np.concatenate((np.zeros(n, dtype=np.bool_), bs))

def bs_extend(bs, n):
    #input bs 1010 output bs 1010...0000 (to match delay lengths)
    return np.concatenate((bs, np.zeros(n, dtype=np.bool_)))

#MACROS
def array_loop(inner, iters, print_result=True):
    vals = np.empty(iters)
    for i in range(iters):
        vals[i] = inner()
    if print_result:
        print("vals: ", vals)
        print("mean: ", np.mean(vals))
        print("std: ", np.std(vals))
    return vals

def avg_loop(inner, iters, print_result=True):
    avg = 0.0
    for _ in range(iters):
        avg += inner()
    if print_result:
        print("avg: ", avg)
    return avg / iters    