import numpy as np

bv_int_cache = {}
def bit_vec_to_int(vec):
    """Utility function for converting a np array bit vector to an integer"""
    str_vec = "".join([str(x) for x in vec])
    if str_vec in bv_int_cache.keys():
        return bv_int_cache[str_vec]
    result = vec.dot(2**np.arange(vec.size)[::-1])
    bv_int_cache[str_vec] = result 
    return result