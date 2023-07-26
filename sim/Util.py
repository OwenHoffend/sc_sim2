import numpy as np

def clog2(N):
    return np.ceil(np.log2(N)).astype(np.int32)

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(
        list(np.binary_repr(num).zfill(m))
    ).astype(bool)[::-1] #Reverse here forces idx 0 to be LSB

bv_int_cache = {}
def bit_vec_to_int(vec):
    """Utility function for converting a np array bit vector to an integer"""
    str_vec = "".join([str(x) for x in vec])
    if str_vec in bv_int_cache.keys():
        return bv_int_cache[str_vec]
    result = vec.dot(2**np.arange(vec.size))
    bv_int_cache[str_vec] = result 
    return result

def bit_vec_arr_to_int(arr):
    _, N = arr.shape
    result = np.zeros(N, dtype=np.int32)
    for i in range(N):
        result[i] = bit_vec_to_int(arr[:, i])
    return result