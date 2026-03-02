import numpy as np
import sympy as sp
from sim.SCC import ascc_from_bs, scc_mat
from sim.SNG import *
from sim.PTV import get_C_from_v, get_Q, get_actual_PTV, get_PTV
from symb_analysis.seq_CAP import get_dv_from_rho_single

def test_autocorr_bitstreams():

    px = 0.5
    w = 16
    N = 2 ** w
    
    #Functions to generate bitstreams with -1, 0, and 1 auto-SCC
    bsx_rho1 = COUNTER_SNG(w, np.eye(1)).run(px, N)
    bsx_rho0 = RAND_SNG(w, np.eye(1)).run(px, N)
    bsx_rho_n1 = MIN_AUTOCORR_SNG(w, np.eye(1)).run(px, N)

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
    dv_actual = get_actual_PTV(bsx_rho, delay=1)
    dv_actual_nrho = get_actual_PTV(bsx_nrho, delay=1)
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

def test_autocorr_bitstream_pair():
    px = 0.5
    py = 0.5
    w = 3
    N = 2 ** (2*w)

    #bs_mat = RAND_SNG(w, np.ones((2, 2))).run([px, py], N)
    #print(scc_mat(bs_mat, delay=1))
    #actual_ptv = get_actual_PTV(bs_mat, delay=1)
    #print(actual_ptv)
#
    ##Compare this to the predicted analytical PTV
    #C = np.array([
    #    [1, 0, 1, 0],
    #    [0, 1, 0, 1],
    #    [1, 0, 1, 0],
    #    [0, 1, 0, 1],
    #])
    #v = get_PTV(C, np.array([px, px, py, py]))
    #print(v)
    #print(np.allclose(actual_ptv, v, atol=1e-2))
#
    ##Now consider the case where multiple bitstreams are generated using the same counter RNS
    #bs_mat = COUNTER_SNG(w, np.eye(2)).run([px, py], N)
    #print(scc_mat(bs_mat, delay=1))
    #actual_ptv = get_actual_PTV(bs_mat, delay=1)
    #print(actual_ptv)
#
    ##Compare this to the predicted analytical PTV
    #C = np.array([
    #    [1, 1, 0, 0],
    #    [1, 1, 0, 0],
    #    [0, 0, 1, 1],
    #    [0, 0, 1, 1],
    #])
    #v = get_PTV(C, np.array([px, px, py, py]))
    #print(v)
    #print(np.allclose(actual_ptv, v, atol=1e-2))
#
    ##Now let's try some cases with negative autocorrelation
    bs_mat = VAN_DER_CORPUT_SNG(w, np.eye(2)).run([px, py], N)
    print(scc_mat(bs_mat, delay=1))
    actual_ptv = get_actual_PTV(bs_mat, delay=1)
    print(actual_ptv)
#
    ##Compare this to the predicted analytical PTV
    C = np.array([
        [1, -1, 0, 0],
        [-1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 0, -1, 1],
    ])
    v = get_PTV(C, np.array([px, px, py, py]))
    print(v)
    print(np.allclose(actual_ptv, v, atol=1e-2))

    #Now let's try both positive and negative autocorrelation
    bs1 = COUNTER_SNG(w, np.eye(1)).run(px, N)
    #bs2 = MIN_AUTOCORR_SNG(w, np.eye(1)).run(py, N)
    bs2 = VAN_DER_CORPUT_SNG(w, np.eye(1)).run(py, N)
    bs_mat = np.concatenate((bs1, bs2), axis=0)
    print(scc_mat(bs_mat, delay=1))
    actual_ptv = get_actual_PTV(bs_mat, delay=1)
    print(actual_ptv)

    #Compare this to the predicted analytical PTV
    C = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 0, -1, 1],
    ])
    v = get_PTV(C, np.array([px, px, py, py]))
    print(v)
    print(np.allclose(actual_ptv, v, atol=1e-2))

def test_autocorr_sng():
    px = 0.25
    w = 12
    rho = -0.5 #can't be exactly 1 because then the output NEVER transitions
    dv = get_dv_from_rho_single(rho).subs("x", px)
    print(dv)
    x1 = dv[2] / (1-px)
    x2 = dv[3] / px

    N = 2 ** (2 * w)
    bs_mat = RAND_SNG(w, np.ones((2, 2))).run([x1, x2], N)
    z_mat = C_AUTOCORR_GEN().run(bs_mat)
    print(np.mean(z_mat))
    print(ascc_from_bs(z_mat))

def test_autocorr_sng_pair():
    px = 0.5
    py = 0.5
    w = 11
    rho_x = 0.99
    #dv_x = get_dv_from_rho_single(rho_x).subs("x", px)
    #x1 = dv_x[2] / (1-px)
    #x2 = dv_x[3] / px

    rho_y = 0
    #dv_y = get_dv_from_rho_single(rho_y).subs("x", py)
    #y1 = dv_y[2] / (1-py)
    #y2 = dv_y[3] / py

    N = 2 ** (2 * w)
    C = 0
    bs_out = RAND_SNG(w, np.array([
        [1, C],
        [C, 1]
    ])).run([px, py], N, rhos=[rho_x, rho_y])
    #bs_mat = RAND_SNG(w, 
    #    np.array([
    #        [1, 1, C, C],
    #        [1, 1, C, C],
    #        [C, C, 1, 1],
    #        [C, C, 1, 1],
    #    ])
    #).run([x1, x2, y1, y2], N)
    #x_out = np.expand_dims(C_AUTOCORR_GEN().run(bs_mat[:2, :]), axis=0)
    #y_out = np.expand_dims(C_AUTOCORR_GEN().run(bs_mat[2:, :]), axis=0)
    #bs_out = np.concatenate((x_out, y_out), axis=0)
    print(np.mean(bs_out, axis=1))
    C_out = scc_mat(bs_out, delay=1)
    print(C_out)

    C_pred = np.array([
        [1, rho_x, C, 0],
        [rho_x, 1, 0, C],
        [C, 0, 1, rho_y],
        [0, C, rho_y, 1],
    ])
    print(np.isclose(C_out, C_pred, atol=0.1))

def test_get_DV_pair_unsatisfiable():
    #consider two variables x and y and their delayed counterparts: Xt, Xt-1, Yt, Yt-1
    #suppose the correlation matrix is:
    #[1, 1, 0, 0]
    #[1, 1, 0, 0]
    #[0, 0, 1, 1]
    #[0, 0, 1, 1]
    #Using the CAP procedure, the p vector is:
    x, y = sp.symbols('x y', real=True, nonneg=True)
    p = sp.Matrix([
        1, 
        y, 
        y, 
        y,
        x,
        x*y,
        x*y,
        x*y,
        x,
        x*y,
        x*y,
        x*y,
        x,
        x*y,
        x*y,
        x*y
    ])
    Q = get_Q(4)
    Q_inv = sp.Matrix(np.linalg.inv(Q))
    v = sp.nsimplify(Q_inv @ p)
    print(sum(v))

    print(get_C_from_v(np.array(v.subs([(x, 0.75), (y, 0.25)]))))