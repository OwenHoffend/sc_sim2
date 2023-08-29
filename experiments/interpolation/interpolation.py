import numpy as np
from scipy import interpolate
from sim.circs import mux, maj
from sim.Util import clog2, MSE
from sim.SNG import lfsr_sng
from sim.SCC import *
from sim.PCC import pcc
from sim.RNS import lfsr
from sim.COMAX import COMAX
import matplotlib.pyplot as plt

def L2_1D(p0, p1, s):
    return (1-s) * p0 + s * p1

def L2_1D_SC(p0, p1, s):
    return mux(s, p0, p1)

def C4_1D(pn1, p0, p1, p2, s):
    Dx_0 = p0 - pn1
    Dx_1 = p1 - p2
    l2_1d = L2_1D(p0, p1, s)
    return l2_1d + (1-s) * s * L2_1D(Dx_0, Dx_1, s)

def C4_1D_SC_L1(pn1, p0, p1, p2, s, s_1ms):
    a = mux(s, p0, p1)
    b = mux(s, pn1, p2)
    l1_top = np.bitwise_and(a, s_1ms)
    l1_bot = np.bitwise_and(b, s_1ms)
    return a, l1_top, l1_bot

def C4_1D_SC_L2(a, l1_top, l1_bot, c1, c2):
    #only one of Da and Db is non-zero
    Da = np.bitwise_and(l1_top, np.bitwise_not(l1_bot))
    Db = np.bitwise_and(l1_bot, np.bitwise_not(l1_top)) 
    aDa = mux(c1, Da, a)
    Db_half = np.bitwise_and(c1, Db)
    return aDa, Db_half

def C4_1D_SC_L3(aDa, Db_half):
    return np.bitwise_and(aDa, np.bitwise_not(Db_half))

def C4_alternate(pn1, p0, p1, p2, s, s_1ms, c1):
    a_p = np.bitwise_and(p0, np.bitwise_not(pn1))
    a_n = np.bitwise_and(pn1, np.bitwise_not(p0)) 

    b_p = np.bitwise_and(p1, np.bitwise_not(p2))
    b_n = np.bitwise_and(p2, np.bitwise_not(p1))

    m_p = mux(s, a_p, b_p)
    m_n = mux(s, a_n, b_n)

    H_p = np.bitwise_and(m_p, s_1ms)
    H_n = np.bitwise_and(m_n, s_1ms)

    L = mux(s, p0, p1)

    c_p = mux(c1, H_p, L)
    c_n = np.bitwise_and(H_n, c1)

    return c_p, c_n

def C4_1D_SC(pn1, p0, p1, p2, s, s_1ms, c1, c2):
    #Layer 1:
    a, l1_top, l1_bot = C4_1D_SC_L1(pn1, p0, p1, p2, s, s_1ms)

    #Layer 2:
    aDa, Db_half =  C4_1D_SC_L2(a, l1_top, l1_bot, c1, c2)

    #Layer 3:
    return C4_1D_SC_L3(aDa, Db_half)

def C4_1D_with_pccs(*x, precision=2, s=0.5, s_1ms=0.25):
    nc = 2 * precision + 1
    xc = x[:nc]
    xv = x[nc:] #indices: [-1, 0, 1, 2]

    s_bit = pcc(xc[:precision], s, precision)
    s_1ms_bit = pcc(xc[precision:2*precision], s_1ms, precision)
    return C4_alternate(xv[0], xv[1], xv[2], xv[3], s_bit, s_1ms_bit, xc[-1])

def full_linear_1d(num_points, upscale_factor, y, SC=False, N=256):
    yinterp = np.zeros((num_points * upscale_factor))
    for idx in range(num_points):
        p0 = y[idx]
        p1 = y[idx+1]
        for s_idx, s in enumerate([i / upscale_factor for i in range(upscale_factor)]):
            if SC:
                w = clog2(N)
                parr_p = np.array([p0, p1])
                bs_mat_p = lfsr_sng(parr_p, N, w, corr=False)
                bs_mat_s = lfsr_sng(np.array((s, )), N, w)
                bs_out = L2_1D_SC(bs_mat_p[0, :], bs_mat_p[1, :], bs_mat_s[0, :])
                yinterp[idx * upscale_factor + s_idx] = np.mean(np.unpackbits(bs_out))
            else:
                yinterp[idx * upscale_factor + s_idx] = L2_1D(p0, p1, s)
    return yinterp

def full_cubic_1d(num_points, upscale_factor, y, SC=False, reco=False, comax=False, precision=5, N=256):
    yinterp = np.zeros((num_points * upscale_factor))
    nc = 2 * precision + 1
    nv = 4

    for idx in range(num_points):
        p0 = y[idx]
        if idx == 0:
            pn1 = p0
        else:
            pn1 = y[idx-1]
        p1 = y[idx+1]
        if idx == num_points-1:
            p2 = p1
        else:
            p2 = y[idx+2]

        for s_idx, s in enumerate([i / upscale_factor for i in range(upscale_factor)]):
            if SC:
                if s_idx == 0:
                    yinterp[idx * upscale_factor] = p0
                    continue
                
                w = clog2(N)
                parr_p = np.array([pn1, p0, p1, p2])
                bs_mat_p = lfsr_sng(parr_p, N, w, corr=True)

                bs_mat_c = np.packbits(lfsr(nc, N), axis=1)
                sc_func = lambda *x : C4_1D_with_pccs(*x, precision=precision, s=s, s_1ms=s*(1-s))

                if comax:
                    sc_func = COMAX(sc_func, nc, nv, 2)

                c_p, c_n = sc_func(*([bs_mat_c[i, :] for i in range(nc)] + [bs_mat_p[i, :] for i in range(4)]))

                #bs_mat_s = lfsr_sng(np.array([s, (1-s) * s, 0.5, 0.5]), N, w)

                #Original method
                #Layer 1:
                #a, l1_top, l1_bot = C4_1D_SC_L1(*([bs_mat_p[i, :] for i in range(4)] + [bs_mat_s[i, :] for i in range(2)]))
                #if reco:
                #    l1_top, l1_bot = reco_2(l1_top, l1_bot)
                #print(scc(l1_top, l1_bot))

                #Layer 2:
                #aDa, Db_half =  C4_1D_SC_L2(a, l1_top, l1_bot, bs_mat_s[2, :], bs_mat_s[3, :])
                #if reco:
                #    aDa, Db_half = reco_2(aDa, Db_half)
                #print(scc(aDa, Db_half))

                #Layer 3:
                #bs_out = C4_1D_SC_L3(aDa, Db_half)

                #Alternate method
                #c_p, c_n = C4_alternate(*([bs_mat_p[i, :] for i in range(4)] + [bs_mat_s[i, :] for i in range(3)]))

                if reco:
                    c_p, c_n = reco_2(c_p, c_n)
                bs_out = C4_1D_SC_L3(c_p, c_n)
                yinterp[idx * upscale_factor + s_idx] = 2 * np.mean(np.unpackbits(bs_out)) #Multiply by 2 because layer 2 of the cubic circuit cuts the value in half
            else:
                yinterp[idx * upscale_factor + s_idx] = C4_1D(pn1, p0, p1, p2, s)
    return yinterp

def test_interp_1d(func, num_points, upscale_factor):
    N = 2048

    num_interps = num_points * upscale_factor
    x = np.linspace(0, 10, num_points+1)
    y = func(x)
    xvals = np.linspace(0, 10, num_interps+1)[:-1]

    yinterp_linear = full_linear_1d(num_points, upscale_factor, y)
    yinterp_linear_SC = full_linear_1d(num_points, upscale_factor, y, SC=True, N=N)
    yinterp_cubic = full_cubic_1d(num_points, upscale_factor, y)
    yinterp_cubic_SC = full_cubic_1d(num_points, upscale_factor, y, SC=True, reco=True, N=N)

    #yinterp_func = interpolate.interp1d(x, y, kind='cubic')
    #yinterp = np.zeros((num_interps))
    #for idx, xval in enumerate(xvals):
    #    yinterp[idx] = yinterp_func(xval)

    correct = func(xvals)
    #print("SciPy MSE: {}".format(np.mean((yinterp - correct) ** 2)))
    print("Linear MSE: {}".format(np.mean((yinterp_linear - correct) ** 2)))
    print("Linear SC MSE: {}".format(np.mean((yinterp_linear_SC - correct) ** 2)))
    print("Cubic MSE: {}".format(np.mean((yinterp_cubic - correct) ** 2)))
    print("Cubic SC MSE: {}".format(np.mean((yinterp_cubic_SC - correct) ** 2)))
    
    plt.plot(x, y, 'o', markersize=9, label="data points")
    #plt.plot(xvals, yinterp, '-x')
    plt.plot(xvals, yinterp_linear_SC, label="linear, SC")
    plt.plot(xvals, yinterp_linear, label="linear, ideal")
    plt.plot(xvals, yinterp_cubic_SC, label="cubic, SC")
    plt.plot(xvals, yinterp_cubic, label="cubic, ideal")
    #plt.plot(xvals, correct, '-^', label="correct")
    plt.title("Linear & Cubic Interpolation, SC vs. Float")
    plt.legend()
    plt.show()

def test_cubic_interp_1d_MSE(func, num_points, upscale_factor, Ns, trials):
    num_interps = num_points * upscale_factor
    x = np.linspace(0, 10, num_points+1)
    y = func(x)
    xvals = np.linspace(0, 10, num_interps+1)[:-1]
    correct = func(xvals)

    mses_SC = np.zeros(len(Ns))
    mses_SC_reco = np.zeros(len(Ns))
    yinterp_cubic = full_cubic_1d(num_points, upscale_factor, y)
    ideal_cubic_mse = MSE(yinterp_cubic, correct)
    print(np.log(ideal_cubic_mse))
    mses_ideal = np.array([ideal_cubic_mse for _ in range(len(Ns))])
    for idx, N in enumerate(Ns):
        for _ in range(trials[idx]):
            yinterp_cubic_SC = full_cubic_1d(num_points, upscale_factor, y, SC=True, N=N)
            yinterp_cubic_SC_reco = full_cubic_1d(num_points, upscale_factor, y, SC=True, reco=True, N=N)
            mses_SC[idx] += MSE(yinterp_cubic_SC, correct)
            mses_SC_reco[idx] += MSE(yinterp_cubic_SC_reco, correct)
        mses_SC[idx] /= trials[idx]
        mses_SC_reco[idx] /= trials[idx]
        print(N)
        print("No reco: {}".format(np.log(mses_SC[idx])))
        print("Yes reco: {}".format(np.log(mses_SC_reco[idx])))

    plt.title("Cubic interp. MSEs vs. SN length, for 0.5*(cos(x) + 1)")
    plt.xlabel("# of bits (N)")
    plt.ylabel("MSE (log scale)")
    plt.plot(Ns, np.log(mses_ideal), label="ideal cubic")
    plt.plot(Ns, np.log(mses_SC), label="SC, no re-corr")
    plt.plot(Ns, np.log(mses_SC_reco), label="SC, re-corr")
    plt.legend()
    plt.show()