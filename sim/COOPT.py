import numpy as np
from sim.circs.circs import Circ
from sim.PTM import *
from sim.PTV import *
from synth.sat import *

#Basic COOPT algorithm without area optimization
def COOPT_via_PTVs(circ: Circ, Cout):
    #First check if the correlation matrix is satisfiable
    m, _ = Cout.shape
    assert circ.m == m

    #first check satisfiability
    sat_result = sat(Cout)
    if sat_result is None:
        print("Desired correlation matrix is not satisfiable")
        return

    #Get the weight matrix
    ptm = get_PTM(circ, lsb='right')
    TT = ptm @ B_mat(circ.m, lsb='right') # 2**n x m
    W = np.zeros((2**circ.nv, circ.m), dtype=np.int32)
    for i in range(circ.m):
        W[:, i] = np.sum(TT[:, i].reshape(2**circ.nv, 2**circ.nc), axis=1)

    Pw = W / 2 ** circ.nc

    #Determine the number of extra constant inputs required
    row_ptvs = np.empty((2 ** circ.nv, 2 ** m))
    for i in range(2 ** circ.nv):
        row_ptvs[i, :] = get_PTV(Cout, Pw[i, :], lsb='right')

    #There's probably a more intelligent algorithm than this, but it works
    num_new_consts = 0
    row_ptv_ints = row_ptvs * (2 ** circ.nc)
    while not np.all(np.round(row_ptv_ints) == row_ptv_ints):
        num_new_consts += 1
        row_ptv_ints = row_ptvs * (2 ** (circ.nc + num_new_consts))

    print("Num new consts required: ", num_new_consts)
    row_ptv_ints = row_ptv_ints.astype(np.int32)

    #Generate a PTV for each row based on the weight probabilities
    #from this, sample the exact bit sequences required for each row
    #TODO: This implementation will likely benefit from caching
    SEMs = np.empty((circ.m, 2 ** circ.nv, 2 ** (circ.nc + num_new_consts)), dtype=np.bool_)
    Bm = B_mat(m, lsb='right')
    for i in range(2 ** circ.nv):
        row_ptv_int = row_ptv_ints[i, :]
        row_idx = 0
        for j in range(2 ** m):
            num = row_ptv_int[j]
            if num == 0:
                continue
            seq = Bm[j]
            tiled = np.tile(seq, (num, 1))
            SEMs[:, i, row_idx:row_idx+num] = tiled.T
            row_idx += num
    return SEMs