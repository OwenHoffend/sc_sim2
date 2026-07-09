import numpy as np
import graycode
from itertools import permutations
from synth.COOPT import *
from synth.espresso import *
from math import factorial
from sim.PTM import get_SEMs_from_ptm
from math import comb
from synth.experiments.example_circuits_for_proposal import Example_Circ_COMAX, Example_Circ_COMAX_OPT_FOR_SCC_0
from sympy.utilities.iterables import multiset_permutations
from synth.branch_and_bound_multi_output import *

def opt_SEM_area_col_perm(Ks):
    #Attempts to optimize the area of the SEM by permuting the columns of the SEM
    #All columns are permuted together, which means that this is not an exhaustive brute-force search
    min_cost = np.inf
    min_Ks = Ks
    min_Mf = Ks_to_Mf(Ks)
    n = len(Ks)
    nv2, nc2 = Ks[0].shape
    num_perms = factorial(nc2)
    max_iters = 5000

    espresso_get_opt_file(min_Mf, "joint_area_example_overall_in.txt", "joint_area_example_overall_out.txt", do_print=True)

    for i, perm in enumerate(permutations(range(nc2))):
        Ks_perm = [Ks[i][:, perm] for i in range(n)] #Naiive column-wise permutation
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

def get_unique_row_patterns(row):
    #for a row of the row_ptv_ints matrix, get the unique ways we can permute the row
    num_patterns = row.size
    nc2 = np.sum(row)
    arr = np.zeros(nc2, dtype=np.int32)
    j = 0
    for i in range(num_patterns):
        num_ith_pattern = row[i]
        for k in range(num_ith_pattern):
            arr[j] = i
            j += 1
            
    for perm in multiset_permutations(arr):
        yield perm

def get_unique_block_row_patterns(row):
    #Unique ways to permute the row assuming that the groups of patterns are contiguous
    num_patterns = row.size
    nc2 = np.sum(row)
    for perm in permutations(range(num_patterns)):
        arr = np.zeros(nc2, dtype=np.int32)
        j = 0
        for i in perm:
            num_ith_pattern = row[i]
            for _ in range(num_ith_pattern):
                arr[j] = i
                j += 1
        yield arr

def get_unique_designs_rec(row_ptv_ints):
    #Iterate through all possible row patterns - this is extremely slow
    for row_pattern in get_unique_block_row_patterns(row_ptv_ints[0, :]):
    #for row_pattern in get_unique_row_patterns(row_ptv_ints[0, :]):
        if row_ptv_ints.shape[0] == 1:
            yield row_pattern
        else:  
            for subpattern in get_unique_designs_rec(row_ptv_ints[1:, :]):
                yield np.vstack((row_pattern, subpattern))

def design_to_Ks_and_Mf(design, nv2, nc2, m, use_gray_code=False):
    Ks = [np.zeros((nv2, nc2), dtype=np.bool_) for _ in range(m)]
    global patterns #remember this so we don't keep calculating it every time
    patterns = B_mat(m, lsb='right')
    if use_gray_code:
        global gc_indices
        gc_indices = np.array(graycode.gen_gray_codes(clog2(nc2)))
    for i in range(nv2):
        for j in range(nc2):
            pattern = patterns[design[i, j]]
            for output in range(m):
                Ks[output][i, j] = pattern[output]
    if use_gray_code:
        Ks = [Ks[output][:, gc_indices] for output in range(m)]
    return Ks, Ks_to_Mf(Ks)

def joint_COOPT_area_brute_force(row_ptv_ints, use_gray_code=False):
    #Rows represent each unique pattern of Xv variable inputs
    #Columns represent the different unique patterns of outputs
    nv2 = row_ptv_ints.shape[0]
    nc2 = np.sum(row_ptv_ints[0, :])
    m = clog2(row_ptv_ints.shape[1])

    min_cost = np.inf
    min_Mf = None
    min_Ks = None
    num_designs = 0
    for design in get_unique_designs_rec(row_ptv_ints):
        #Convert the design to SEMs
        Ks, Mf = design_to_Ks_and_Mf(design, nv2, nc2, m, use_gray_code)

        cost = espresso_get_SOP_area(Mf, "joint_area_example.txt")
        if cost < min_cost:
            min_cost = cost
            min_Mf = Mf
            min_Ks = Ks
            print(f"Current min: {min_cost}")

        num_designs += 1
        if num_designs % 1000 == 0:
            print(f"Num designs: {num_designs}")

    print(f"Min cost: {min_cost}")
    print("Min Ks: ")
    for i in range(m):
        print(min_Ks[i])

def mutate_design(Ks, swap_chance=0.1):
    Ks_copy = [K.copy() for K in Ks]
    for row in range(Ks[0].shape[0]):
        for col in range(Ks[0].shape[1]): 
            if np.random.random() < swap_chance:
                swap_dest = np.random.randint(0, Ks[0].shape[1])
                for i in range(len(Ks)):
                    Ks_copy[i][row, col], Ks_copy[i][row, swap_dest] = Ks_copy[i][row, swap_dest], Ks_copy[i][row, col]
    return Ks_copy

def joint_COOPT_area_genetic(Ks, num_gens = 1000, pop_size = 100):
    min_cost = np.inf
    min_Ks = None
    m = len(Ks)

    #First initialize a random population
    population = [mutate_design(Ks) for _ in range(pop_size)]
    min_costs = []
    for gen in range(num_gens):
        print(f"Generation {gen}/{num_gens}")
        #Evaluate the fitness of the population

        fitnesses = []
        for Ks in population:
            Mf = Ks_to_Mf(Ks)
            cost = espresso_get_SOP_area(Mf, "joint_area_example.txt")
            if cost < min_cost:
                min_cost = cost
                min_Ks = Ks
                print(f"Current min: {min_cost}")
            fitnesses.append(cost)

        print(f"Mean fitness: {np.mean(fitnesses)}")
        print(f"Min fitness: {np.min(fitnesses)}")
        min_costs.append(np.min(fitnesses))
        #Select the fittest individuals
        population = [population[i] for i in np.argsort(fitnesses)][:pop_size//5]
        #Create the next generation
        population = [mutate_design(population[i % pop_size//4]) for i in range(pop_size - pop_size//5)] + population

    print(f"Min cost: {min_cost}")
    print("Min Ks: ")
    for i in range(m):
        print(min_Ks[i])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(min_costs)
    plt.xlabel("Generation")
    plt.ylabel("Minimum Cost")
    plt.title("Minimum Cost per Generation")
    plt.tight_layout()
    plt.show()

def joint_COOPT_area_brute_force_each_row(row_ptv_ints, use_gray_code=False):
    #Brute force optimization applied to each row individually instead of for all rows at once
    nv2 = row_ptv_ints.shape[0]
    nc2 = np.sum(row_ptv_ints[0, :])
    m = clog2(row_ptv_ints.shape[1])
    global patterns #remember this so we don't keep calculating it every time
    patterns = B_mat(m, lsb='right')

    Ks = [np.zeros((nv2, nc2), dtype=np.bool_) for _ in range(m)]
    for i in range(nv2):
        min_row_cost = np.inf
        min_row = None
        print(f"Row {i}/{nv2}")
        #for pattern in get_unique_row_patterns(row_ptv_ints[i, :]): #this is computationally expensive
        for pattern in get_unique_block_row_patterns(row_ptv_ints[i, :]):
            Ks_row = [np.zeros((1, nc2), dtype=np.bool_) for _ in range(m)]
            for j in range(nc2):
                current_pattern = patterns[pattern[j]]
                for output in range(m):
                    Ks_row[output][0, j] = current_pattern[output]

            if use_gray_code:
                gc_indices = np.array(graycode.gen_gray_codes(clog2(nc2)))
                Ks_row = [Ks_row[output][:, gc_indices] for output in range(m)]
            Mf = Ks_to_Mf(Ks_row)
            cost = espresso_get_SOP_area(Mf, "joint_area_example.txt")
            if cost < min_row_cost:
                print(f"New min row cost: {cost}")
                min_row_cost = cost
                min_row = Ks_row
        print(f"Min row cost: {min_row_cost}")
        for j in range(m):
            Ks[j][i, :] = min_row[j][0, :]

    Mf = Ks_to_Mf(Ks)
    cost = espresso_get_SOP_area(Mf, "joint_area_example.txt")
    print(f"Total cost: {cost}")
    print(Ks)

def greedy_gray_code_opt(row_ptv_ints):
    #First reoganize the row_ptv_ints into gray-code order
    nv2 = row_ptv_ints.shape[0]
    nc2 = np.sum(row_ptv_ints[0, :])
    m = clog2(row_ptv_ints.shape[1])
    gray_code_col_indices = np.array(graycode.gen_gray_codes(m))
    gray_code_row_indices = np.array(graycode.gen_gray_codes(clog2(nv2)))
    row_ptv_ints = row_ptv_ints[gray_code_row_indices, :]
    row_ptv_ints = row_ptv_ints[:, gray_code_col_indices]

    #Not finished because I am abandoning this approach for now
    pass

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

    #opt_SEM_area_col_perm([K1_opt, K2_opt])

def joint_area_example_DV_based():

    circ_opt = Example_Circ_COMAX_OPT_FOR_SCC_0()
    Mf_opt = circ_opt.get_PTM()
    cost_opt = espresso_get_SOP_area(Mf_opt, "joint_area_example_opt.txt")
    Ks_opt = get_SEMs_from_ptm(Mf_opt, circ_opt.m, circ_opt.nc, circ_opt.nv)
    print(cost_opt)
    #print(Ks_opt)
    print(get_row_MVs_from_SEMs(Ks_opt))

    #DV-based COOPT implementation
    circ = Example_Circ_COMAX()
    #Cout = np.ones((circ.m, circ.m))
    Cout = np.identity(circ.m)
    circ_opt = COOPT_via_PTVs(circ, Cout)
    Ks_opt = get_SEMs_from_ptm(circ_opt.get_PTM(), circ_opt.m, circ_opt.nc, circ_opt.nv)
    #for K in Ks_opt:
    #    print(K)

    print(get_row_MVs_from_SEMs(Ks_opt))

    row_ptv_ints = COOPT_via_PTVs(circ, Cout, return_only_row_DVs=True)
    Ks_opt = branch_and_bound_opt_multi_output(row_ptv_ints)
    #for K in Ks_opt:
    #    print(K)

    print(get_row_MVs_from_SEMs(Ks_opt))

    cost_opt = espresso_get_SOP_area(Ks_to_Mf(Ks_opt), "joint_area_example.txt")
    print(cost_opt)