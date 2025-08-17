import unittest
import numpy as np
from synth.sat import *
from sim.PTV import *

correct_table_3x3 = {
    "000": True,
    "001": True,
    "00-1": True,
    "010": True,
    "011": False,
    "01-1": False,
    "0-10": True,
    "0-11": False,
    "0-1-1": False,
    "100": True,
    "101": False,
    "10-1": False,
    "110": False,
    "111": True,
    "11-1": False,
    "1-10": False,
    "1-11": False,
    "1-1-1": True,
    "-100": True,
    "-101": False,
    "-10-1": False,
    "-110": False,
    "-111": False,
    "-11-1": True,
    "-1-10": False,
    "-1-11": True,
    "-1-1-1": False
}

class TestSat(unittest.TestCase):
    def test_all_3x3(self):
        print("test_all_3x3")
        for c1 in [0, -1, 1]:
            for c2 in [0, -1, 1]:
                for c3 in [0, -1, 1]:
                    C = np.array([
                        [1, c1, c2],
                        [c1, 1, c3],
                        [c2, c3, 1]
                    ])
                    sat_result = sat(C)
                    sat_axiom_result = sat_via_axioms(C)
                    sat_PSD_result = sat_via_PSD(C)
                    key = f"{c1}{c2}{c3}"
                    self.assertEqual(sat_result is not None, correct_table_3x3[key], f"C= {key}")
                    self.assertEqual(sat_axiom_result, correct_table_3x3[key], f"C= {key}")
                    self.assertEqual(sat_PSD_result, correct_table_3x3[key], f"C= {key}")
        print("PASS")

    def test_all_4x4(self):
        print("test_all_4x4")
        for c1 in [0, -1, 1]: #lazy programming but it doesn't matter
            for c2 in [0, -1, 1]:
                for c3 in [0, -1, 1]:
                    for c4 in [0, -1, 1]:
                        for c5 in [0, -1, 1]:
                            for c6 in [0, -1, 1]:
                                C = np.array([
                                    [1, c1, c2, c3],
                                    [c1, 1, c4, c5],
                                    [c2, c4, 1, c6],
                                    [c3, c5, c6, 1],
                                ])
                                sat_result = sat(C)
                                sat_axiom_result = sat_via_axioms(C)
                                sat_PSD_result = sat_via_PSD(C)
                                self.assertEqual(sat_result is not None, sat_axiom_result)
                                self.assertEqual(sat_axiom_result, sat_PSD_result)

                                Px = np.array([0.5, 0.5, 0.5, 0.5])
                                v = get_PTV(C, Px)
                                if v is None:
                                    continue
                                C_out = get_C_from_v(v)
                                P_out = get_Px_from_v(v)
                                self.assertTrue(np.all(np.isclose(C_out, C)), f"Correlation \n{C}, {Px}")
                                self.assertTrue(np.all(np.isclose(P_out, Px)), f"Probability \n{C}, {Px}")

        print("PASS")

    def test_random_ptv(self): 
        print("test_random_ptv")
        ns = [3, 4, 5, 6]
        num_tests = 10000

        for n in ns:
            print(f"n: {n}")
            total_sat = 0
            for t in range(num_tests):
                #get random correlation matrix
                C = np.identity(n)
                for i in range(n):
                    for j in range(i):
                        val = np.random.choice([-1, 0, 1])
                        C[i, j] = val
                        C[j, i] = val

                #get random probabilities
                Px = np.random.uniform(size=(n,))

                #check satisfiability
                sat_axiom_result = sat_via_axioms(C)
                sat_PSD_result = sat_via_PSD(C)
                self.assertEqual(sat_axiom_result, sat_PSD_result)

                v = get_PTV(C, Px)
                if v is None:
                    continue
                total_sat += 1
                C_out = get_C_from_v(v)
                P_out = get_Px_from_v(v)
                self.assertTrue(np.all(np.isclose(C_out, C)), f"Correlation \n{C}, {Px}")
                self.assertTrue(np.all(np.isclose(P_out, Px)), f"Probability \n{C}, {Px}")
            print(f"total sat: {total_sat}/{num_tests} for n={n}")
        print("PASS")