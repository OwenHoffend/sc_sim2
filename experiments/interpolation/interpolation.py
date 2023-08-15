import numpy as np
from scipy import interpolate
from sim.circs import mux
from sim.Util import clog2
from sim.SNG import lfsr_sng
import matplotlib.pyplot as plt

def L2_1D(p0, p1, s):
    return (1-s) * p0 + s * p1

def L2_1D_SC(p0, p1, s):
    return mux(s, p0, p1)

def Q3_1D(p0, p1, p_half, s):
    D_half = p_half - (p0 + p1) / 2
    l2_1d = L2_1D(p0, p1, s)
    return l2_1d + 4 * (1-s) * s * D_half

#def C4_1D(pn1, p0, p1, p2, s):
#    Px_0 = p1 - pn1
#    Px_1 = -(p2 - p0)
#    Dx_0 = (Px_0 - (p1 - p0))
#    Dx_1 = (Px_1 - (p0 - p1))
#    l2_1d = L2_1D(p0, p1, s)
#    return l2_1d + (1-s) * s * L2_1D(Dx_0, Dx_1, s)

def C4_1D(Px_0, Px_1, p0, p1, s):
    Dx_0 = Px_0 - (p1 - p0)
    Dx_1 = -Px_1 - (p0 - p1)
    l2_1d = L2_1D(p0, p1, s)
    return l2_1d + (1-s) * s * L2_1D(Dx_0, Dx_1, s)

def full_linear_1d(num_points, upscale_factor, y):
    yinterp = np.zeros((num_points * upscale_factor))
    for idx in range(num_points):
        p0 = y[idx]
        p1 = y[idx+1]
        for s_idx, s in enumerate([i / upscale_factor for i in range(upscale_factor)]):
            yinterp[idx * upscale_factor + s_idx] = L2_1D(p0, p1, s)
    return yinterp

def full_linear_1d_SC(num_points, upscale_factor, y, N=256):
    yinterp = np.zeros((num_points * upscale_factor))
    for idx in range(num_points):
        p0 = y[idx]
        p1 = y[idx+1]
        for s_idx, s in enumerate([i / upscale_factor for i in range(upscale_factor)]):
            w = clog2(N)
            parr_p = np.array([p0, p1])
            bs_mat_p = lfsr_sng(parr_p, N, w, corr=False)
            bs_mat_s = lfsr_sng(np.array((s, )), N, w)
            bs_out = L2_1D_SC(bs_mat_p[0, :], bs_mat_p[1, :], bs_mat_s[0, :])
            yinterp[idx * upscale_factor + s_idx] = np.mean(np.unpackbits(bs_out))
    return yinterp

def full_quad_1d(num_points, upscale_factor, y):
    yinterp = np.zeros((num_points * upscale_factor))
    #srange = [i / (2*upscale_factor) for i in range(2*upscale_factor)]
    #last_range = np.zeros((upscale_factor))
    #for idx in range(num_points - 2):
    #    p0 = y[idx]
    #    p_half = y[idx+1]
    #    p1 = y[idx+2]
    #    for s_idx, s in enumerate(srange):
    #        yinterp[idx * upscale_factor + s_idx] = Q3_1D(p0, p1, p_half, s)
    #    if idx != 0:
    #        yinterp[idx * upscale_factor : (idx + 1) * upscale_factor] /= 2 
    #        yinterp[idx * upscale_factor : (idx + 1) * upscale_factor] += last_range / 2
    #    last_range = yinterp[(idx + 1) * upscale_factor : (idx + 2) * upscale_factor]
    
    for idx in range(np.floor(num_points/2).astype(np.int32)):
        p0 = y[idx*2]
        p_half = y[idx*2+1]
        p1 = y[idx*2+2]
        for s_idx, s in enumerate([i / (2*upscale_factor) for i in range(2*upscale_factor)]):
            yinterp[idx * 2 * upscale_factor + s_idx] = Q3_1D(p0, p1, p_half, s)
    
    return yinterp

def full_cubic_1d(num_points, upscale_factor, y):
    yinterp = np.zeros((num_points * upscale_factor))

    delta_prev = 0.0
    for idx in range(num_points):
        p0 = y[idx]
        p1 = y[idx+1]
        delta = (y[idx + 1] - y[idx])
        if idx == 0:
            Px_0 = delta
        else:
            Px_0 = delta_prev

        Px_1 = delta
        delta_prev = delta

        #p0 = y[idx]
        #if idx == 0:
        #    pn1 = p0
        #else:
        #    pn1 = y[idx-1]
        #p1 = y[idx+1]
        #if idx == num_points-2:
        #    p2 = p1
        #else:
        #    p2 = y[idx+2]

        for s_idx, s in enumerate([i / upscale_factor for i in range(upscale_factor)]):
            yinterp[idx * upscale_factor + s_idx] = C4_1D(Px_0, Px_1, p0, p1, s)
            #yinterp[idx * upscale_factor + s_idx] = C4_1D(pn1, p0, p1, p2, s)
    return yinterp

def test_interp_1d(func, kind, num_points, upscale_factor):

    num_interps = num_points * upscale_factor
    x = np.linspace(0, 10, num_points+1)
    y = func(x)
    xvals = np.linspace(0, 10, num_interps+1)[:-1]
    yinterp_func = interpolate.interp1d(x, y, kind=kind)
    yinterp = np.zeros((num_interps))

    yinterp_linear = full_linear_1d(num_points, upscale_factor, y)
    yinterp_linear_sc = full_linear_1d_SC(num_points, upscale_factor, y)
    yinterp_cubic = full_cubic_1d(num_points, upscale_factor, y)

    for idx, xval in enumerate(xvals):
        yinterp[idx] = yinterp_func(xval)

    correct = func(xvals)
    print("SciPy MSE: {}".format(np.mean((yinterp - correct) ** 2)))
    print("Linear MSE: {}".format(np.mean((yinterp_linear - correct) ** 2)))
    print("Linear SC MSE: {}".format(np.mean((yinterp_linear_sc - correct) ** 2)))
    print("Cubic MSE: {}".format(np.mean((yinterp_cubic - correct) ** 2)))
    
    plt.plot(x, y, 'o', markersize=9, label="data points")
    #plt.plot(xvals, yinterp, '-x')
    #plt.plot(xvals, yinterp_cubic, '-v')
    plt.plot(xvals, yinterp_linear_sc, label="linear, SC")
    plt.plot(xvals, yinterp_linear, label="linear, ideal")
    #plt.plot(xvals, correct, '-^')
    plt.legend()
    plt.show()
