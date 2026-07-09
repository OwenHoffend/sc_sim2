import numpy as np
import copy
from sim.Util import clog2
from sim.PTV import get_Q
from functools import reduce

class Cube:
    def __init__(self, n: int, mask: int, bits: int):
        self.n = n #total number of variables
        self.mask = mask # 1 means the variable is specified
        self.bits = bits # values of the specified variables

        #Compute size
        inv_mask = ~self.mask & ((1 << self.n) - 1)
        self.size = 1 << (inv_mask).bit_count()

        #Compute minterms covered
        #This gives a vector like 11001100 (for n=3) showing which minterms this cube covers
        cover_mask = 0
        for i in range(self.size):
            minterm_idx = scatter_bits(i, inv_mask) ^ self.bits
            cover_mask |= 1 << minterm_idx
        self.cover_mask = cover_mask
        self.cover_bool_array = mask_to_bool_array(cover_mask, 1 << self.n)

    def covers(self, point: int) -> bool:
        return (self.cover_mask & point) == self.cover_mask

    def __str__(self) -> str:
        return f"Mask: {bin(self.mask)}, Bits: {bin(self.bits)}, size: {self.size}, cover_mask: {bin(self.cover_mask)}"

class CubePairMulti:
    def __init__(self, vcube: Cube, ccube: Cube, output_mask: int):
        self.vcube = vcube
        self.ccube = ccube
        self.output_mask = output_mask #indicates which outputs this cube connects to
        self.score = vcube.size * ccube.size
        self.lit_count = vcube.mask.bit_count() + ccube.mask.bit_count()
        self.weight_update = vcube.cover_bool_array * ccube.size

    def overlaps(self, other: "CubePairMulti") -> bool:
        return (self.vcube.cover_mask & other.vcube.cover_mask) != 0 \
            and (self.ccube.cover_mask & other.ccube.cover_mask) != 0 \
            and (self.output_mask & other.output_mask) != 0 #allow overlapping cubes if they connect to different outputs

    def __str__(self) -> str:
        return f"vcube: {self.vcube}, ccube: {self.ccube}, score: {self.score}, weight_update: {self.weight_update}, output_mask: {self.output_mask}"

def convert_cube_pairs_to_SEMs(cube_pairs: list[CubePairMulti], nv: int, nc: int, m: int) -> list[np.ndarray]:
    SEMs: list[np.ndarray] = [np.zeros((2 ** nv, 2 ** nc), dtype=np.bool_) for _ in range(m)]
    for cube in cube_pairs:
        vmask = 1
        for v_idx in range(2 ** nv):
            if vmask & cube.vcube.cover_mask:
                cmask = 1
                for c_idx in range(2 ** nc):
                    if cmask & cube.ccube.cover_mask:
                        omask = 1
                        for o_idx in range(m):
                            if omask & cube.output_mask:
                                SEMs[o_idx][v_idx, c_idx] = True
                            omask <<= 1
                    cmask <<= 1
            vmask <<= 1

    return SEMs

class ProblemVector:
    def __init__(self, p: np.ndarray, output_mask: int):
        self.p = p
        self.output_mask = output_mask
        self.sz = output_mask.bit_count()

def get_problem_matrix_from_DV(V: np.ndarray) -> list[ProblemVector]:
    nv = clog2(V.shape[0])
    m = clog2(V.shape[1])

    #first convert into a matrix of p (marginal) vectors
    Q = get_Q(m, lsb='right')
    P = np.empty((2 ** nv, 2 ** m))
    for i in range(2 ** nv):
        P[i, :] = Q @ V[i, :]

    #order by number of outputs each marginal involves
    P_vecs: list[ProblemVector] = []
    for i in range(1, 2 ** m): #skip the 0 index, as this doesn't represent any actual output
        P_vecs.append(ProblemVector(P[:, i], i))

    return P_vecs

class Node:
    def __init__(self, pm: list[ProblemVector], cube_set: list[CubePairMulti], m: int):
        self.m = m
        self.pm = pm
        self.cube_set = cube_set
        self.lit_count = sum([cs.lit_count for cs in cube_set]) if cube_set != [] else 0
        self.current_output_mask = get_largest_unsolved_output_mask(pm, m)

    def overlaps(self, other: CubePairMulti) -> bool:
        if self.cube_set == []:
            return False
        for pair in self.cube_set:
            if pair.overlaps(other):
                return True
        return False

    def is_solved(self) -> bool:
        return self.current_output_mask == -1
        #for p in self.pm:
        #    if np.any(p != 0):
        #        return False
        #return True

def scatter_bits(x: int, target_mask: int) -> int:
    """
    Scatter the low-order bits of x into the 1-bit positions of target_mask.

    Example:
        x = 0b101
        target_mask = 0b10110

        result has x's bits placed into positions 1, 2, and 4.
    """
    result = 0
    src_bit = 1

    while target_mask:
        dst_bit = target_mask & -target_mask  # lowest set bit in target_mask

        if x & src_bit:
            result |= dst_bit

        src_bit <<= 1
        target_mask ^= dst_bit

    return result

def reverse_bits(x: int, n: int) -> int:
    """
    Reverse the lowest n bits of x.

    Example:
        x = 0b1101, n = 4 -> 0b1011
    """
    result = 0

    for _ in range(n):
        result = (result << 1) | (x & 1)
        x >>= 1

    return result

def get_unique_cubes(n: int) -> list[Cube]:
    cubes = []
    for mask in range(1 << n):
        num_cubes = 1 << mask.bit_count() #number of cubes with this mask
        for cube in range(num_cubes):
            bits = scatter_bits(cube, mask)
            new_cube = Cube(n, mask, bits)
            cubes.append(new_cube)
            #print(new_cube)
    return cubes

def bool_array_to_mask(arr: np.ndarray) -> int:
    #Note that this uses indexing order: [False, True, True, True] -> 1110, because the "False" is the 0th index
    arr = np.asarray(arr, dtype=bool).ravel()

    mask = 0
    for i, value in enumerate(arr):
        if value:
            mask |= 1 << i

    return mask

def mask_to_bool_array(mask: int, n:int) -> np.ndarray:
    arr = np.zeros(n, dtype=bool)
    for i in range(n):
        if mask & (1 << i):
            arr[i] = True
    return arr

def count_in_bitcount_order(bit_width: int):
    counts = []
    for i in range(1, 1 << bit_width):
        counts.append((i.bit_count(), i))
    counts.sort(key=lambda x: -x[0])
    return [x[1] for x in counts]

def get_largest_unsolved_output_mask(pm: list[ProblemVector], m: int):

    #FIXME: always choosing a specific solve order may be overly restrictive
    #Consider allowing cases like 110, 101, 011 which do not conflict to be solved in any order
    for mask in count_in_bitcount_order(m):
        m_rev = reverse_bits(mask, m)
        if np.any(pm[m_rev - 1].p > 0):
            return m_rev

    return -1 #indicates the problem is solved

def find_largest_valid_cubes(node: Node, vcubes: list[Cube], ccubes_by_size: dict) -> list[CubePairMulti]:
    """Given a weight vector, find all of the valid cubes
    that do not intersect with an existing set of cubes. Order these by size

    The assumption is that the weight vector has already been updated to reflect the impact of the existing cube set
    """

    w = node.pm[node.current_output_mask - 1].p

    #For each cube of the variable inputs, there is a "multiplicity" which is the smallest weight covered by the vcube
    max_w = np.max(w)
    max_multiplicity = 1 << (np.floor(np.log2(max_w)).astype(int))
    #print(f"max_multiplicity: {max_multiplicity}")

    vcubes_under_consideration = copy.deepcopy(vcubes)

    largest_cubes: list[CubePairMulti] = []
    largest_score = 0
    current_multiplicity = 1
    while current_multiplicity <= max_multiplicity:
        #print(f"current_multiplicity: {current_multiplicity}")
        proj = bool_array_to_mask(w >= current_multiplicity) #These are all the weights that have at least this multiplicity
        new_vcubes = []
        for vcube in vcubes_under_consideration:
            if vcube.covers(proj):
                new_vcubes.append(vcube)
                #print(f"vcube: {vcube}")
                for ccube in ccubes_by_size[current_multiplicity]:
                    cube_pair = CubePairMulti(vcube, ccube, node.current_output_mask)
                    if cube_pair.score < largest_score:
                        continue
                    elif not node.overlaps(cube_pair):
                        if cube_pair.score > largest_score:
                            largest_cubes = [cube_pair, ]
                            largest_score = cube_pair.score
                        else: #Equal to the largest score
                            largest_cubes.append(cube_pair)
        vcubes_under_consideration = new_vcubes
        current_multiplicity <<= 1

    return largest_cubes

#This is the part of the code that corresponds directly to [Qian, 2017]
def branch_and_bound_opt_multi_output(V: np.ndarray) -> Node:
    nv = clog2(V.shape[0])
    nc = clog2(np.sum(V[0, :]))
    m = clog2(V.shape[1])

    initial_pm = get_problem_matrix_from_DV(V)

    N = Node(initial_pm, [], m)
    N_best = Node(initial_pm, [], m)
    n0 = np.inf
    stk = [N, ]

    vcubes = get_unique_cubes(nv)
    ccubes = get_unique_cubes(nc)

    ccubes_by_size = {}
    for ccube in ccubes:
        sz = ccube.size
        if sz not in ccubes_by_size:
            ccubes_by_size[sz] = [ccube,]
        else:
            ccubes_by_size[sz].append(ccube)

    iter_ctr = 0
    while stk != []:
        if iter_ctr % 100 == 0:
            print(f"Iteration: {iter_ctr}, stack size: {len(stk)}")
        iter_ctr += 1
        N = stk.pop()
        L = find_largest_valid_cubes(N, vcubes, ccubes_by_size)
        while len(L) > 0:
            C = L.pop()
            if N.lit_count + C.lit_count < n0:
                new_pm: list[ProblemVector] = []
                for pv in N.pm:
                    if (pv.output_mask & C.output_mask) == pv.output_mask:
                        new_p = pv.p - C.weight_update
                    if np.any(new_p < 0):
                        raise ValueError("Found an invalid cube assignment")
                    new_pm.append(ProblemVector(new_p, pv.sz))
                cube_set_new = N.cube_set + [C, ]
                N_new = Node(new_pm, cube_set_new, m)

                if N_new.is_solved(): #reached a leaf node
                    n0 = N_new.lit_count
                    N_best = N_new
                    print(f"New best: {n0} at iteration {iter_ctr}")
                else:
                    stk.append(N_new)

    #print the result
    for cube in N_best.cube_set:
        print(cube)
    print(n0)
    Ks = convert_cube_pairs_to_SEMs(N_best.cube_set, nv, nc, m)
    for K in Ks:
        print(K)
    return N_best
