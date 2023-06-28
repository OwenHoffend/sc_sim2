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