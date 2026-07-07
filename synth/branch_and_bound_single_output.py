import numpy as np
import copy
import heapq
from itertools import count
from sim.Util import clog2

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

class CubePair:
    def __init__(self, vcube: Cube, ccube: Cube):
        self.vcube = vcube
        self.ccube = ccube
        self.score = vcube.size * ccube.size #number of terms this cube pair covers
        self.lit_count = vcube.mask.bit_count() + ccube.mask.bit_count()
        self.weight_update = vcube.cover_bool_array * ccube.size

    def overlaps(self, other: "CubePair") -> bool:
        return (self.vcube.cover_mask & other.vcube.cover_mask) != 0 \
            and (self.ccube.cover_mask & other.ccube.cover_mask) != 0

    def __str__(self) -> str:
        return f"vcube: {self.vcube}, ccube: {self.ccube}, score: {self.score}, weight_update: {self.weight_update}"

class Node:
    def __init__(self, w: np.ndarray, cube_set: list[CubePair]):
        self.w = w
        self.cube_set = cube_set
        self.lit_count = sum([cs.lit_count for cs in cube_set]) if cube_set != [] else 0

    def overlaps(self, other: CubePair) -> bool:
        if self.cube_set == []:
            return False
        for pair in self.cube_set:
            if pair.overlaps(other):
                return True
        return False

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

def find_largest_valid_cubes(node: Node, vcubes: list[Cube], ccubes_by_size: dict) -> list[CubePair]:
    """Given a weight vector, find all of the valid cubes
    that do not intersect with an existing set of cubes. Order these by size

    The assumption is that the weight vector has already been updated to reflect the impact of the existing cube set
    """

    #For each cube of the variable inputs, there is a "multiplicity" which is the smallest weight covered by the vcube
    max_w = np.max(node.w)
    max_multiplicity = 1 << (np.floor(np.log2(max_w)).astype(int))
    #print(f"max_multiplicity: {max_multiplicity}")
    current_multiplicity = 1

    vcubes_under_consideration = copy.deepcopy(vcubes)

    largest_cubes: list[CubePair] = []
    largest_score = 0
    while current_multiplicity <= max_multiplicity:
        #print(f"current_multiplicity: {current_multiplicity}")
        proj = bool_array_to_mask(node.w >= current_multiplicity) #These are all the weights that have at least this multiplicity
        new_vcubes = []
        for vcube in vcubes_under_consideration:
            if vcube.covers(proj):
                new_vcubes.append(vcube)
                #print(f"vcube: {vcube}")
                for ccube in ccubes_by_size[current_multiplicity]:
                    cube_pair = CubePair(vcube, ccube)
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
def branch_and_bound_opt_single_output(w: np.ndarray) -> Node:
    nv = clog2(w.size)
    nc = clog2(np.max(w))
    N = Node(w, [])
    N_best = Node(w, [])
    n0 = np.inf
    stk = [N, ]

    vcubes = get_unique_cubes(nv)
    ccubes = get_unique_cubes(nc)

    ccubes_by_size = {}
    for ccube in ccubes:
        sz = ccube.size
        if sz not in ccubes_by_size:
            ccubes_by_size[ccube.size] = [ccube,]
        else:
            ccubes_by_size[ccube.size].append(ccube)

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
                w_new = N.w - C.weight_update
                if np.any(w_new < 0):
                    raise ValueError("Weight update is negative")
                cube_set_new = N.cube_set + [C, ]
                N_new = Node(w_new, cube_set_new)
                if np.all(w_new == 0): #reached a leaf node
                    n0 = N_new.lit_count
                    N_best = N_new
                    print(f"New best: {n0} at iteration {iter_ctr}")
                else:
                    stk.append(N_new)

    #print the result
    for cube in N_best.cube_set:
        print(cube)
    print(n0)
    return N_best
