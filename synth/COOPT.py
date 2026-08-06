import numpy as np
from sim.Util import int_array
from sim.circs.circs import Circ, PTM_Circ
from synth.sat import *
from sim.PTM import *
from sim.PTV import *

#Basic COOPT algorithm without area optimization
def COOPT_via_PTVs(
    circ: Circ, 
    Cout, 
    return_only_row_DVs=False, 
    return_only_Ks=False, 
    approx_higher_order_marginals=False
):
    #First check if the correlation matrix is satisfiable
    m, _ = Cout.shape
    assert circ.m == m

    #first check satisfiability
    #sat_result = sat(Cout)
    sat_result = sat_via_PSD(Cout)
    if sat_result is None:
        print("Desired correlation matrix is not satisfiable")
        return

    #Get the weight matrix
    #TODO: use the get_SEMs_from_ptm functions instead, at some point
    ptm = circ.get_PTM(lsb='right')
    TT = ptm @ B_mat(circ.m, lsb='right') # 2**n x m
    W = np.zeros((2**circ.nv, circ.m), dtype=np.int32)
    for i in range(circ.m):
        W[:, i] = np.sum(TT[:, i].reshape(2**circ.nv, 2**circ.nc), axis=1)

    Pw = W / 2 ** circ.nc

    row_mvs = np.empty((2 ** circ.nv, 2 ** m))

    #we only care about having exact precision on MV entries representing 1st and 2nd order marginals 
    row_mvs_masked = np.empty((2 ** circ.nv, 2 ** m))
    for i in range(2 ** circ.nv):
        row_mvs[i, :] = get_PTV(Cout, Pw[i, :], lsb='right', mv_only=True)
        #row_ptvs[i, :] = get_DV_via_gaussian_copula(Cout, Pw[i, :], lsb='right').round(12)
        row_mvs_masked[i, :] = row_mvs[i, :]

    #apply mask to row_mvs_masked
    if approx_higher_order_marginals:
        for k, Dk, in enumerate(get_D(m)):
            #maximum marginal size to consider, 
            # higher marginals are rounded to the precision of the lower ones
            if len(Dk) > 2: 
                row_mvs_masked[:, k] = 0

    #Determine the number of constant inputs required
    #There's probably a more intelligent algorithm than this, but it works
    num_required_consts = 0
    row_mvs_int = row_mvs_masked * (2 ** num_required_consts)
    err_thresh = 1e-3
    while not np.allclose(np.round(row_mvs_int), row_mvs_int, atol=err_thresh):
        num_required_consts += 1
        row_mvs_int = row_mvs_masked * 2 ** num_required_consts
    print("Num consts required: ", num_required_consts)

    #After determining the correct precision to not lose probability/correlation information
    #approximate the other marginal probabilities at the reduced precision
    #FIXME: May need to check that the rounded MVs still obey the FH bounds
    row_mvs_rounded = round_prec(row_mvs, num_required_consts)
    Q = get_Q(m, lsb='right')
    Q_inv = np.linalg.inv(Q)
    
    row_dvs = np.zeros_like(row_mvs_rounded)
    for i in range(2 ** circ.nv):
        row_dvs[i, :] = Q_inv @ row_mvs_rounded[i, :]

        #ensure stochastic equivalence and correlation optimality
        assert np.all(get_Px_from_v(row_dvs[i, :]) == Pw[i, :])
        assert np.all(get_C_from_v(row_dvs[i, :]) == Cout)
        assert np.sum(row_dvs[i, :]) == 1
        assert np.all(0 <= row_dvs[i, :]) and np.all(row_dvs[i, :] <= 1)

    row_dv_ints = (row_dvs * 2 ** num_required_consts).astype(np.int32)

    if return_only_row_DVs:
        #dividing this by 2 ** num_required_consts will give the fully reduced circuit PTM
        return row_dv_ints

    #From here, we just build a naiive SEM matrix from the row DVs
    #The joint area optimization would theoretically be integrated here

    #Generate a DV for each row based on the weight probabilities
    #from this, sample the exact bit sequences required for each row
    #TODO: This implementation will likely benefit from caching
    SEMs = np.empty((circ.m, 2 ** circ.nv, 2 ** num_required_consts), dtype=np.bool_)
    Bm = B_mat(m, lsb='right')
    for i in range(2 ** circ.nv):
        row_dv_int = row_dv_ints[i, :]
        row_idx = 0
        for j in range(2 ** m):
            num = row_dv_int[j]
            if num == 0:
                continue
            seq = Bm[j]
            tiled = np.tile(seq, (num, 1))
            SEMs[:, i, row_idx:row_idx+num] = tiled.T
            row_idx += num

    if return_only_Ks:
        return SEMs

    Mf_opt = Ks_to_Mf(SEMs)
    circ_opt = PTM_Circ(Mf_opt, circ)
    circ_opt.nc = num_required_consts
    
    return circ_opt

#Legacy implementation of COMAX
def opt_K_max(K):
    _, tlen = K.shape
    K_sum = np.sum(K, axis=1).astype(np.int32)
    return np.stack([np.pad(np.ones(t, dtype=np.bool_), (0, tlen-t), 'constant') for t in K_sum])

def A_to_Mf(A, n, k):
    Mf = np.zeros((2**n, 2**k), dtype=np.bool_)
    for i in range(2**n):
        x = int_array(A[i, :].reshape(1, k), lsb='right')[0]
        Mf[i, x] = True
    return Mf

def Ks_to_Mf(Ks):
    nc, nv = np.log2(Ks[0].shape)
    n = int(nv + nc)
    k = len(Ks) #Ks is a list of K matrices
    A = Ks_to_A(Ks)
    return A_to_Mf(A, n, k)

def Ks_to_A(Ks):
    nc, nv = np.log2(Ks[0].shape)
    n = int(nv + nc)
    k = len(Ks) #Ks is a list of K matrices
    A = np.zeros((2**n, k), dtype=np.bool_)
    for i, K in enumerate(Ks):
        A[:, i] = K.reshape(2**n)
    return A