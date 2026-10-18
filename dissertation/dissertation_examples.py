from sim.sim import *
from sim.SNG import *
from sim.PCC import *
from sim.SCC import *
from sim.RNS import *
from sim.datasets import *
from sim.circs.circs import *
from CAP.PTV import *
from CAP.PTM import *
import matplotlib.pyplot as plt
from experiments.example_circuits_for_proposal import *

def scc_norm(pxy, x, y):
    if(pxy > x*y):
        return min(x, y) - x*y
    else:
        return x*y - max(x+y-1, 0)

def xor_and_example_sccfunc_case1(x1, x2, x3):
    y1y2 = xor_and_example_model_prediction_case1(x1, x2, x3)
    y1 = abs(x1 - x2)
    y2 = x3
    return (y1y2 - y1*y2) / scc_norm(y1y2, y1, y2)

def xor_and_example_model_prediction_case1(x1, x2, x3):
    return abs(x1 - x2) * x3

#Derived via PTM-based correlation analysis
def xor_and_example_sccfunc_case2(x1, x2, x3):
    y1y2 = xor_and_example_model_prediction_case2(x1, x2, x3)
    y1 = abs(x1 - x2)
    y2 = x3
    return (y1y2 - y1*y2) / scc_norm(y1y2, y1, y2)

#Derived via PTM-based correlation analysis
def xor_and_example_model_prediction_case2(x1, x2, x3):
    return max(0, min(x2, x3) - x1) + max(0, min(x1, x3) - x2)

def xor_and_example_dissertation():

    w = 8

    x1 = 0.75
    x2 = 0.25

    Cin = np.ones((3, 3))
    circ = XOR_with_AND(Cin=Cin)
    ds = dataset_all_same(1000, 1, 0.75)
    ds = ds.merge(dataset_all_same(ds.num, 1, 0.25))
    ds = ds.merge(dataset_sweep_1d(ds.num)) #x1, x2, x3 = [0.75, 0.25, x3]
    sng = LFSR_SNG(w, Cin, nc=circ.nc)
    result = sim_circ(sng, circ, ds)

    xvals = dataset_sweep_1d(ds.num).ds
    sccs_correct_case1 = np.zeros_like(xvals)
    vals_correct_case1 = np.zeros_like(xvals)
    sccs_correct_case2 = np.zeros_like(xvals)
    vals_correct_case2 = np.zeros_like(xvals)
    for i, xval in enumerate(xvals):
        sccs_correct_case1[i] = xor_and_example_sccfunc_case1(x1, x2, xvals[i])
        vals_correct_case1[i] = xor_and_example_model_prediction_case1(x1, x2, xvals[i])
        sccs_correct_case2[i] = xor_and_example_sccfunc_case2(x1, x2, xvals[i])
        vals_correct_case2[i] = xor_and_example_model_prediction_case2(x1, x2, xvals[i])

    # Create one figure with four subplots (1x2 grid)
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(8, 3))

    # First subplot - SCCs for correlated version
    sccs_corr = [scc for scc in circ.internal_sccs]
    ax1.plot(xvals[5:], sccs_corr[5:], label="Actual SCC")
    #ax1.plot(xvals[5:], sccs_correct_case2[5:], 'r--', label="CAP prediction")
    ax1.set_xlabel(r"$P_{X_3}$", fontsize=12)
    ax1.set_ylabel(r"$SCC(Y_1, Y_2)$", fontsize=12)
    ax1.set_title(r"$(a)$ $SCC(Y_1, Y_2)$ vs. $P_{X_3}$", fontsize=12)
    ax1.legend()

    # Second subplot - Output values
    ax2.plot(xvals, result.out, label="Actual Function") #case where AND gate implements MIN
    ax2.plot(xvals, result.correct, label=r"$min(|P_{X_1}-P_{X_2}|, P_{X_3})$")
    ax2.plot(xvals, vals_correct_case1, label=r"$|P_{X_1}-P_{X_2}|P_{X_3}$") #case where AND gate implements MUL
    ax2.set_xlabel(r"$P_{X_3}$", fontsize=12)
    ax2.set_ylabel(r"$P_{Z}$", fontsize=12)
    ax2.set_title(r"$(b)$ $P_{Z}$ vs. $P_{X_3}$", fontsize=12)
    ax2.legend()

    plt.tight_layout(pad=0.5)
    plt.show()