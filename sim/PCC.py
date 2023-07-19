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