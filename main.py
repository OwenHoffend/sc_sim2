from sim.SNG import *
from sim.PCC import *
from sim.RNS import *

if __name__ == "__main__":
    nv = 2
    nc = 1
    w = 4
    cgroups = [0, 1]
    sng = LFSR_SNG_WN(nv, nc, w, cgroups)
    bs_mat = sng.run([])
