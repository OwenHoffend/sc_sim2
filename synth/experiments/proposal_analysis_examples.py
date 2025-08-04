from sim.sim import *
from sim.SNG import *
from sim.PCC import *
from sim.SCC import *
from sim.RNS import *
from sim.datasets import *
from sim.circs.circs import *
from sim.PTV import *
from sim.PTM import *
import matplotlib.pyplot as plt
from synth.experiments.example_circuits_for_proposal import *

def scc_norm(pxy, x, y):
    if(pxy > x*y):
        return min(x, y) - x*y
    else:
        return x*y - max(x+y-1, 0)

#Derived via PTM-based correlation analysis
def xor_and_example_sccfunc(x1, x2, x3):
    y1y2 = max(0, min(x2, x3) - x1) + max(0, min(x1, x3) - x2)
    y1 = abs(x1 - x2)
    y2 = x3
    return (y1y2 - y1*y2) / scc_norm(y1y2, y1, y2)

#Derived via PTM-based correlation analysis
def xor_and_example_model_prediction(x1, x2, x3):
    return max(0, min(x2, x3) - x1) + max(0, min(x1, x3) - x2)

def xor_and_example():

    x1 = 0.75
    x2 = 0.25

    circ = XOR_with_AND()
    ds = dataset_all_same(1000, 1, 0.75)
    ds = ds.merge(dataset_all_same(ds.num, 1, 0.25))
    ds = ds.merge(dataset_sweep_1d(ds.num)) #x1, x2, x3 = [0.75, 0.25, x3]
    sng = LFSR_SNG(8, circ)
    result = sim_circ(sng, circ, ds)

    xvals = dataset_sweep_1d(ds.num).ds
    sccs_correct_case2 = np.zeros_like(xvals)
    vals_correct_case2 = np.zeros_like(xvals)
    for i, xval in enumerate(xvals):
        sccs_correct_case2[i] = xor_and_example_sccfunc(x1, x2, xvals[i])
        vals_correct_case2[i] = xor_and_example_model_prediction(x1, x2, xvals[i])

    #mul_vals = np.array([0.5 * x for x in xvals])
    #min_vals = np.array([np.minimum(x, 0.5) for x in xvals])
    #neg_vals = np.array([np.maximum(0.5 + x - 1,0) for x in xvals])
#
    #scc_predicted_values = []
    #for i in range(len(xvals)):
    #    C = circ.internal_sccs[i]
    #    if circ.internal_sccs[i] < 0:
    #        scc_predicted_values.append((1+C)*mul_vals[i] - C * neg_vals[i])
    #    else:
    #        scc_predicted_values.append((1-C)*mul_vals[i] + C * min_vals[i])

    # Run experiment with uncorrelated version
    circ_uncorr = XOR_with_AND_uncorr()
    sng_uncorr = LFSR_SNG(6, circ_uncorr)
    result_uncorr = sim_circ(sng_uncorr, circ_uncorr, ds)

    # Create one figure with four subplots (2x2 grid)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
    
    # First subplot - uncorrelated version
    ax1.plot(xvals, result_uncorr.out, label="Output")
    ax1.plot(xvals, result_uncorr.correct, label="Correct")
    ax1.set_xlabel(r"$P_{X_3}$", fontsize=12)
    ax1.set_ylabel(r"$P_{Z}$", fontsize=12)
    ax1.set_title("Case 1")
    ax1.legend()

    # Second subplot - correlated version
    ax2.plot(xvals, result.out, label="Output")
    ax2.plot(xvals, result.correct, label="Correct")
    ax2.plot(xvals, vals_correct_case2, label="Analysis prediction")
    ax2.set_xlabel(r"$P_{X_3}$", fontsize=12)
    ax2.set_ylabel(r"$P_{Z}$", fontsize=12)
    ax2.set_title("Case 2")
    ax2.legend()

    # Third subplot - SCCs for uncorrelated version
    sccs_uncorr = [scc for scc in circ_uncorr.internal_sccs]
    ax3.plot(xvals[5:], sccs_uncorr[5:], label="SCC")
    ax3.set_xlabel(r"$P_{X_3}$", fontsize=12)
    ax3.set_ylabel("SCC", fontsize=12)
    ax3.set_title("SCCs - Case 1")
    ax3.legend()

    # Fourth subplot - SCCs for correlated version
    sccs_corr = [scc for scc in circ.internal_sccs]
    ax4.plot(xvals[5:], sccs_corr[5:], label="SCC")
    ax4.plot(xvals[5:], sccs_correct_case2[5:], label="Analysis prediction")
    ax4.set_xlabel(r"$P_{X_3}$", fontsize=12)
    ax4.set_ylabel("SCC", fontsize=12)
    ax4.set_title("SCCs - Case 2")
    ax4.legend()

    plt.tight_layout(pad=0.5)
    plt.show()

    # second plot comparing with the predicted SCC curve
