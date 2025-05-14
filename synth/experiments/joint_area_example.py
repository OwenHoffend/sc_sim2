import numpy as np
from itertools import permutations
from synth.COOPT import *
from synth.espresso import *
from math import factorial

def plot_costs_perms(Ks):
    min_cost = np.inf
    min_Ks = Ks
    min_Mf = Ks_to_Mf(Ks)
    n = len(Ks)
    nv2, nc2 = Ks[0].shape
    num_perms = factorial(n)
    max_iters = 500

    espresso_get_opt_file(min_Mf, "joint_area_example_overall_in.txt", "joint_area_example_overall_out.txt", do_print=True)

    for i, perm in enumerate(permutations(range(nc2))):
        Ks_perm = [Ks[i][:, perm] for i in range(n)]
        Mf = Ks_to_Mf(Ks_perm)
        cost = espresso_get_SOP_area(Mf, "joint_area_example.txt")
        if i % 100 == 0:
            print(f"i: {i}/{num_perms}")
        if cost < min_cost:
            min_cost = cost
            min_Ks = Ks_perm
            min_Mf = Ks_to_Mf(Ks_perm)
            print(f"Current min: {min_cost}")
        if i == max_iters:
            break

    espresso_get_opt_file(min_Mf, "joint_area_example_overall_opt_in.txt", "joint_area_example_overall_opt_out.txt", do_print=True)
    print(f"Min cost: {min_cost}")
    print("Min Ks: ")
    for i in range(n):
        print(min_Ks[i])


def joint_area_example():
    #Example of jointly optimizing area and correlation via COMAX, using the SEMs from Fig. 8 of the COMAX paper
    K1 = np.array([
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1]
    ])

    K2 = np.array([
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 1, 1, 1, 1]
    ])
    K1_opt = opt_K_max(K1)
    K2_opt = opt_K_max(K2)
    Mf = Ks_to_Mf([K1, K2])
    Mf_opt = Ks_to_Mf([K1_opt, K2_opt])
    cost = espresso_get_SOP_area(Mf, "joint_area_example.txt")
    cost_opt = espresso_get_SOP_area(Mf_opt, "joint_area_example_opt.txt")
    print(cost)
    print(cost_opt)

    plot_costs_perms([K1_opt, K2_opt])