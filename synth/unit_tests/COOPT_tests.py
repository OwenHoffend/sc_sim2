import unittest
import numpy as np
from sim.sim import *
from sim.PTM import *
from sim.PTM import get_PTM, get_SEMs_from_ptm
from synth.experiments.example_circuits_for_proposal import Example_Circ_COMAX
from synth.COOPT import COOPT_via_PTVs, Ks_to_Mf
from sim.datasets import dataset_uniform

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
        Mf = get_PTM(self.circ)
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
            Mf = get_PTM(self.circ)
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
