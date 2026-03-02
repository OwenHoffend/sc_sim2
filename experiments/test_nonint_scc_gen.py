import numpy as np
from sim.SNG import *
from sim.SCC import scc_mat
from synth.sat import cgroups_and_signs_to_C

def test_nonint_scc_gen_2vars():
    parr = [0.5, 0.5]
    rns_config = [[[-0.5, 0.5], [0, 1]], [[0.7, 0.3], [0, 2]]]
    #in this example, the possible RNS selections are:
        #0.5 * 0.7: SCC = 1
        #0.5 * 0.3: SCC = 0
        #0.5 * 0.7: SCC = 0
        #0.5 * 0.3: SCC = 0
        #So the expected output SCC is -0.35
    cgroups, signs, parr_map = unpack_rns_config(rns_config)
    Cin = cgroups_and_signs_to_C(cgroups, signs)
    print(Cin)

    w = 16
    N = 2 ** w
    bs_mat_r = RAND_SNG(w, Cin).run(parr_map(parr), N)
    bs_mat = mux_for_nonint_SCC(bs_mat_r, rns_config, 0)
    print(np.mean(bs_mat, axis=1))
    print(scc_mat(bs_mat)) #works as of 1/6/2025

def test_nonint_scc_gen_3vars():
    num_tests = 1000
    for i in range(num_tests):
        parr = [np.random.rand(), np.random.rand(), np.random.rand()]
        aconsts = np.array([
            [1, 0, 0],
            [0.6, 0.4, 0],
            [0.1, 0.9, 0]
        ]) #np.random.rand(3, 3)

        row_sums = np.sum(aconsts, axis=1)
        aconsts_norm = aconsts / row_sums[:, np.newaxis]

        #Convert aconsts_norm to a rns_config
        rns_config = []
        for i in range(3):
            rns_config.append([aconsts_norm[i, :], [0, 1, 2]])

        #Compute the expected SCC based on the equation in the 1/15/2026 slides
        C_expected = np.eye(3)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                for k in range(3):
                    C_expected[i, j] += aconsts_norm[i, k] * aconsts_norm[j, k]

        cgroups, signs, parr_map = unpack_rns_config(rns_config)
        Cin = cgroups_and_signs_to_C(cgroups, signs)

        w = 18
        N = 2 ** w
        bs_mat_r = RAND_SNG(w, Cin).run(parr_map(parr), N)
        bs_mat = mux_for_nonint_SCC(bs_mat_r, rns_config, 0)
        print(aconsts_norm)
        print(C_expected)
        print(scc_mat(bs_mat))
        print(np.isclose(scc_mat(bs_mat), C_expected, atol=0.1))
        if not np.allclose(scc_mat(bs_mat), C_expected, atol=0.1):
            print("FAIL")
        print("--------------------------------")

def test_block_mixture_counterexample():
    w = 20
    N = 2 ** w
    parr = [0.5, 0.01, 0.5]
    C_top = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])
    bs_mat_ctop = RAND_SNG(w, C_top).run(parr, N)
    print(scc_mat(bs_mat_ctop))
    print(np.mean(bs_mat_ctop, axis=1))

    C_bot = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])
    bs_mat_cbot = RAND_SNG(w, C_bot).run(parr, N)
    print(scc_mat(bs_mat_cbot))
    print(np.mean(bs_mat_cbot, axis=1))

    bs_mat_mixed = np.zeros_like(bs_mat_cbot)
    for i in range(N):
        r = np.random.rand()
        if r > 0.5:
            bs_mat_mixed[:, i] = bs_mat_ctop[:, i]
        else:
            bs_mat_mixed[:, i] = bs_mat_cbot[:, i]
    print(scc_mat(bs_mat_mixed))
    print(np.mean(bs_mat_mixed, axis=1))
        