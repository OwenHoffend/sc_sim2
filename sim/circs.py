import numpy as np

def mux(s, x, y):
    return np.bitwise_or(
        np.bitwise_and(np.bitwise_not(s), x), 
        np.bitwise_and(s, y)
    )

def maj(s, x, y):
    return np.bitwise_or(
        np.bitwise_or(
            np.bitwise_and(s, x), 
            np.bitwise_and(s, y)
        ),
        np.bitwise_and(x, y)
    )

def robert_cross(s, x11, x22, x12, x21, is_maj=False):
    xor1, xor2 = np.bitwise_xor(x11, x22), np.bitwise_xor(x12, x21)
    if is_maj:
        return maj(s, xor1, xor2)
    else:
        return mux(s, xor1, xor2)