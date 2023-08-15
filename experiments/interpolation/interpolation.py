import numpy as np
from scipy import interpolate
from sim.circs import mux, maj
from sim.Util import clog2
from sim.SNG import lfsr_sng
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

def C4_1D_SC(pn1, p0, p1, p2, s, s_1ms, c1, c2):
    #Layer 1:
    a = mux(s, p0, p1)
    b = mux(s, pn1, p2)
    l1_top = np.bitwise_and(a, s_1ms)
    l1_bot = np.bitwise_and(b, s_1ms)

    #Layer 2:
    #only one of Da and Db is non-zero
    Da = np.bitwise_and(l1_top, np.bitwise_not(l1_bot))
    Db = np.bitwise_and(l1_bot, np.bitwise_not(l1_top)) 
    aDa = mux(c1, Da, a)
    Db_half = np.bitwise_and(c2, Db)

    #Layer 3:
    return np.bitwise_and(aDa, np.bitwise_not(Db_half))

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

def full_cubic_1d(num_points, upscale_factor, y, SC=False, N=256):
    yinterp = np.zeros((num_points * upscale_factor))

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
                w = clog2(N)
                parr_p = np.array([pn1, p0, p1, p2])
                bs_mat_p = lfsr_sng(parr_p, N, w, corr=True)
                bs_mat_s = lfsr_sng(np.array([s, (1-s) * s, 0.5, 0.5]), N, w)
                bs_out = C4_1D_SC(*([bs_mat_p[i, :] for i in range(4)] + [bs_mat_s[i, :] for i in range(4)]))
                yinterp[idx * upscale_factor + s_idx] = 2 * np.mean(np.unpackbits(bs_out)) #Multiply by 2 because layer 2 of the cubic circuit cuts the value in half
            else:
                yinterp[idx * upscale_factor + s_idx] = C4_1D(pn1, p0, p1, p2, s)
    return yinterp

def test_interp_1d(func, kind, num_points, upscale_factor):

    N = 1024

    num_interps = num_points * upscale_factor
    x = np.linspace(0, 10, num_points+1)
    y = func(x)
    xvals = np.linspace(0, 10, num_interps+1)[:-1]
    yinterp_func = interpolate.interp1d(x, y, kind=kind)
    yinterp = np.zeros((num_interps))

    yinterp_linear = full_linear_1d(num_points, upscale_factor, y)
    yinterp_linear_SC = full_linear_1d(num_points, upscale_factor, y, SC=True, N=N)
    yinterp_cubic = np.clip(full_cubic_1d(num_points, upscale_factor, y), 0, 1)
    yinterp_cubic_SC = full_cubic_1d(num_points, upscale_factor, y, SC=True, N=N)

    for idx, xval in enumerate(xvals):
        yinterp[idx] = yinterp_func(xval)

    correct = func(xvals)
    print("SciPy MSE: {}".format(np.mean((yinterp - correct) ** 2)))
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
