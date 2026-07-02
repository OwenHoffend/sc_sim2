import numpy as np
from itertools import permutations
from synth.COOPT import *
from synth.espresso import *
from math import factorial
from sim.PTM import get_SEMs_from_ptm
from math import comb
from synth.experiments.example_circuits_for_proposal import Example_Circ_COMAX
from sympy.utilities.iterables import multiset_permutations

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

def get_num_corr_optimal_equivalent_designs(Ks):
    #First compute the total number of possible equivalent designs that are correlation-optimal
    prod = 1
    nv2, nc2 = Ks[0].shape
    m = len(Ks)
    for i in range(nv2):
        max_comb = 0
        for k in range(m):
            if comb(nc2, Ks[k][i, :].sum()) > max_comb:
                max_comb = comb(nc2, Ks[k][i, :].sum())
        prod *= max_comb
    print(f"Total number of possible equivalent designs: {prod}")

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

def get_unique_designs_rec(row_ptv_ints):
    #Iterate through all possible row patterns - this is extremely slow
    for row_pattern in get_unique_row_patterns(row_ptv_ints[0, :]):
        if row_ptv_ints.shape[0] == 1:
            yield row_pattern
        else:  
            for subpattern in get_unique_designs_rec(row_ptv_ints[1:, :]):
                yield np.vstack((row_pattern, subpattern))

def design_to_Ks_and_Mf(design, nv2, nc2, m):
    Ks = [np.zeros((nv2, nc2), dtype=np.bool_) for _ in range(m)]
    global patterns #remember this so we don't keep calculating it every time
    patterns = B_mat(m, lsb='right')
    for i in range(nv2):
        for j in range(nc2):
            pattern = patterns[design[i, j]]
            for output in range(m):
                Ks[output][i, j] = pattern[output]
    return Ks, Ks_to_Mf(Ks)

def joint_COOPT_area_brute_force(row_ptv_ints):
    #Rows represent each unique pattern of Xv variable inputs
    #Columns represent the different unique patterns of outputs
    nv2 = row_ptv_ints.shape[0]
    nc2 = np.sum(row_ptv_ints[0, :])
    m = clog2(row_ptv_ints.shape[1])

    min_cost = np.inf
    min_Mf = None
    min_Ks = None
    for design in get_unique_designs_rec(row_ptv_ints):
        #Convert the design to SEMs
        Ks, Mf = design_to_Ks_and_Mf(design, nv2, nc2, m)

        cost = espresso_get_SOP_area(Mf, "joint_area_example.txt")
        if cost < min_cost:
            min_cost = cost
            min_Mf = Mf
            min_Ks = Ks
            print(f"Current min: {min_cost}")

    print(f"Min cost: {min_cost}")
    print("Min Ks: ")
    for i in range(m):
        print(min_Ks[i])

def mutate_design(design, swap_chance=0.1):
    design_copy = design.copy()
    for row in range(design.shape[0]):
        for col in range(design.shape[1]): 
            if np.random.random() < swap_chance:
                swap_dest = np.random.randint(0, design.shape[1])
                design_copy[row, col], design_copy[row, swap_dest] = design_copy[row, swap_dest], design_copy[row, col]
    return design_copy

def joint_COOPT_area_genetic(row_ptv_ints, num_gens = 100, pop_size = 100):

    min_cost = np.inf
    min_Mf = None
    min_Ks = None
    nv2 = row_ptv_ints.shape[0]
    nc2 = np.sum(row_ptv_ints[0, :])
    m = clog2(row_ptv_ints.shape[1])

    init_design = next(get_unique_designs_rec(row_ptv_ints))

    #First initialize a random population
    population = [mutate_design(init_design) for _ in range(pop_size)]
    for gen in range(num_gens):
        print(f"Generation {gen}/{num_gens}")
        #Evaluate the fitness of the population

        fitnesses = []
        for design in population:
            Ks, Mf = design_to_Ks_and_Mf(design, nv2, nc2, m)
            cost = espresso_get_SOP_area(Mf, "joint_area_example.txt")
            if cost < min_cost:
                min_cost = cost
                min_Mf = Mf
                min_Ks = Ks
                print(f"Current min: {min_cost}")
            fitnesses.append(cost)

        print(f"Mean fitness: {np.mean(fitnesses)}")
        print(f"Min fitness: {np.min(fitnesses)}")
        #Select the fittest individuals
        population = [population[i] for i in np.argsort(fitnesses)][:pop_size//4]
        #Create the next generation
        population = [mutate_design(population[i % pop_size//4]) for i in range(pop_size//2)] + population[:pop_size//4]

    print(f"Min cost: {min_cost}")
    print("Min Ks: ")
    for i in range(m):
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

    #opt_SEM_area_col_perm([K1_opt, K2_opt])

def joint_area_example_DV_based():
    #DV-based COOPT implementation
    circ = Example_Circ_COMAX()
    #Cout = np.ones((circ.m, circ.m))
    Cout = np.identity(circ.m)
    #circ_opt = COOPT_via_PTVs(circ, Cout)

    row_ptv_ints = COOPT_via_PTVs(circ, Cout, return_only_row_DVs=True)
    print(row_ptv_ints)
    joint_COOPT_area_genetic(row_ptv_ints)

    #Some old code for reference
    #Mf = circ.get_PTM()
    #Mf_opt = circ_opt.get_PTM()
    #cost = espresso_get_SOP_area(Mf, "joint_area_example.txt")
    #cost_opt = espresso_get_SOP_area(Mf_opt, "joint_area_example_opt.txt")
    #print(cost)
    #print(cost_opt)

    #Ks = get_SEMs_from_ptm(Mf_opt, circ_opt.m, circ_opt.nc, circ_opt.nv)
    #opt_SEM_area_col_perm(Ks)