import unittest
import numpy as np
import matplotlib.pyplot as plt
from sim.sim import *
from sim.PTM import *
from sim.PTM import get_SEMs_from_ptm
from synth.experiments.example_circuits_for_proposal import Example_Circ_COMAX, Example_Circ_COOPT
from synth.COOPT import COOPT_via_PTVs, Ks_to_Mf
from sim.datasets import dataset_uniform
from sim.circs.circs import PTM_Circ
from sim.visualization import plot_scc_histogram

class TestCircuitSimulation(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.circ = Example_Circ_COMAX()
        self.Cin = np.ones((self.circ.nv, self.circ.nv)) #Edit this to test different input correlations
        self.Cout = np.ones((self.circ.m, self.circ.m)) #Edit this to test different output correlations
        self.ds = dataset_uniform(1000, self.circ.nv)

    def test_ptm_sem_conversion(self):
        """Test PTM to SEM to PTM conversion maintains circuit behavior."""
        # Get original PTM
        Mf = self.circ.get_PTM(lsb='right')
        ptm_circ = PTM_Circ(Mf, self.circ)

        try:
            sim_result_original = sim_circ_PTM(ptm_circ, self.ds, self.Cin, validate=True)
        except Exception as e:
            self.fail("Original PTM simulation did not match the expected output: " + str(e))

        # Convert to SEMs and back to PTM
        SEMs = get_SEMs_from_ptm(Mf, self.circ.m, self.circ.nc, self.circ.nv)
        Mf_test = Ks_to_Mf(SEMs)
        ptm_circ_test = PTM_Circ(Mf_test, self.circ)

        try:
            sim_result_converted = sim_circ_PTM(ptm_circ_test, self.ds, self.Cin, validate=True)
        except Exception as e:
            self.fail("Converted PTM simulation did not match the expected output: " + str(e))

        # Compare results
        np.testing.assert_array_almost_equal(
            sim_result_original.out, 
            sim_result_converted.out,
            decimal=6,
            err_msg="PTM to SEM to PTM conversion changed circuit behavior"
        )
        print("PASS")

    def test_coopt_optimization(self):
        """Test that COOPT optimization produces valid results."""
        # Get original circuit results
        try:
            Mf = self.circ.get_PTM(lsb='right')
            ptm_circ = PTM_Circ(Mf, self.circ)
            sim_result_original = sim_circ_PTM(ptm_circ, self.ds, self.Cin, validate=True)
        except Exception as e:
            self.fail("Original PTM simulation failed: " + str(e))

        # Get optimized circuit results
        try:
            circ_opt = COOPT_via_PTVs(self.circ, self.Cout)
            sim_result_opt = sim_circ_PTM(circ_opt, self.ds, self.Cin, validate=True)
        except Exception as e:
            self.fail("Optimized circuit simulation failed: " + str(e))

        # Compare results
        np.testing.assert_array_almost_equal(
            sim_result_original.out, 
            sim_result_opt.out,
            decimal=6,
            err_msg="COOPT optimization changed circuit behavior"
        )
        print("PASS")

def COOPT_example_2output(): #TODO convert to unit test
    c = Example_Circ_COMAX()
    ds = dataset_uniform(1000, c.nv)
    Cin = np.ones((c.nv, c.nv))
    #Cin = 0.5 * np.identity(c.m) + 0.5 * (-np.ones((c.m, c.m)) + 2 * np.identity(c.m))
    #Cin = 2 * np.identity(c.nv) - np.ones((c.nv, c.nv))
    Cout = np.ones((c.m, c.m))
    #Cout = 0.5 * np.identity(c.m) + 0.5 * (-np.ones((c.m, c.m)) + 2 * np.identity(c.m))
    #Cout = 2 * np.identity(c.m) - np.ones((c.m, c.m))
    circ_opt = COOPT_via_PTVs(c, Cout)
    sim_result_opt = sim_circ_PTM(circ_opt, ds, Cin, validate=False)
    sim_result_original = sim_circ_PTM(c, ds, Cin, validate=False)
    original_scc_array = sim_result_original.scc_array(0, 1)
    opt_scc_array = sim_result_opt.scc_array(0, 1)

    print("Average SCC of original circuit: ", np.mean(original_scc_array))
    print("Average SCC of optimized circuit: ", np.mean(opt_scc_array))

    #Plot the results
    plot_scc_histogram(
        [original_scc_array, opt_scc_array], 
        labels=["Original", "Optimized"],
        title="COOPT Example Circuit, SCC of Z1 and Z2",
        xlabel="SCC",
        ylabel="Frequency",
        bins=30
    )

def COOPT_example_3output(): #TODO convert to unit test
    c = Example_Circ_COOPT()
    ds = dataset_uniform(1000, c.nv)
    Cin = np.eye(c.nv)
    #Cin = np.ones((c.nv, c.nv))
    Cout = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    #Cout = np.ones((c.m, c.m))
    circ_opt = COOPT_via_PTVs(c, Cout)
    sim_result_opt = sim_circ_PTM(circ_opt, ds, Cin, validate=False)
    sim_result_original = sim_circ_PTM(c, ds, Cin, validate=False)

    # Plot histograms for all pairs of outputs (3 pairs for 3 outputs) as subplots
    pairs = [(0, 1), (0, 2), (1, 2)]
    bins = 30
    colors = plt.cm.tab10.colors[:2]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (out1, out2) in zip(axes, pairs):
        orig_corr = sim_result_original.scc_array(out1, out2)
        opt_corr = sim_result_opt.scc_array(out1, out2)
        for values, label, color in zip(
            [orig_corr, opt_corr], ["Original", "Optimized"], colors
        ):
            ax.hist(values, bins=bins, range=(-1, 1), alpha=0.7, label=label, color=color)
        ax.set_title(f"SCC of Z{out1+1} and Z{out2+1}")
        ax.set_xlabel("SCC")
        ax.set_xlim(-1.1, 1.1)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Frequency")
    fig.suptitle("COOPT Example Circuit, SCC Distributions")
    plt.tight_layout()
    plt.show()