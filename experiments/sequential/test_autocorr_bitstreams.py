import numpy as np
from sim.SCC import ascc_from_bs
from sim.SNG import COUNTER_SNG, RAND_SNG, MIN_AUTOCORR_SNG, nonint_scc
from sim.circs.circs import C_WIRE
from sim.PTV import get_actual_DV_1cycle, get_C_from_v
from symb_analysis.seq_CAP import get_dv_from_rho_single

def test_autocorr_bitstreams():

    px = 0.5
    w = 20
    N = 2 ** w
    
    #Functions to generate bitstreams with -1, 0, and 1 auto-SCC
    bsx_rho1 = COUNTER_SNG(w, C_WIRE(1, np.eye(1))).run(px, N)
    bsx_rho0 = RAND_SNG(w, C_WIRE(1, np.eye(1))).run(px, N)
    bsx_rho_n1 = MIN_AUTOCORR_SNG(w, C_WIRE(1, np.eye(1))).run(px, N)

    #Function to create random combinations of these integer bitstreams
    rho = 0.25
    bsx_rho = nonint_scc(bsx_rho0, bsx_rho1, np.sqrt(rho))
    bsx_nrho = nonint_scc(bsx_rho0, bsx_rho_n1, np.sqrt(rho))

    #Function to measure the autocorrelation of the combined bitstreams
    print(ascc_from_bs(bsx_rho1))
    print(ascc_from_bs(bsx_rho0))
    print(ascc_from_bs(bsx_rho_n1))
    print(ascc_from_bs(bsx_rho))
    print(ascc_from_bs(bsx_nrho))
    #print(np.mean(bsx_rho))
    #print(np.mean(bsx_nrho))

    #Convert the bitstreams into DVs
    dv_actual = get_actual_DV_1cycle(bsx_rho)
    dv_actual_nrho = get_actual_DV_1cycle(bsx_nrho)
    print(dv_actual)
    print(dv_actual_nrho)

    #Get the analytical DV
    dv = get_dv_from_rho_single(rho).subs("x", px)
    dv_nrho = get_dv_from_rho_single(-rho).subs("x", px)
    print(dv)
    print(dv_nrho)

    #Get the actual correlation matrices
    print(get_C_from_v(np.array(dv_actual)))
    print(get_C_from_v(np.array(dv_actual_nrho)))
    print(get_C_from_v(np.array(dv)))
    print(get_C_from_v(np.array(dv_nrho)))