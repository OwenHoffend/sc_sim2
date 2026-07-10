import numpy as np
import copy
from sim.Util import clog2
from sim.PTV import get_Q, get_row_MVs_from_SEMs
import time
import heapq
from itertools import count

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
        self.score = vcube.size * ccube.size * output_mask.bit_count()
        self.lit_count = vcube.mask.bit_count() + ccube.mask.bit_count()
        self.weight_update = vcube.cover_bool_array * ccube.size

    def overlaps(self, other: "CubePairMulti") -> bool:
        return (self.vcube.cover_mask & other.vcube.cover_mask) != 0 \
            and (self.ccube.cover_mask & other.ccube.cover_mask) != 0 \
            and (self.output_mask ^ other.output_mask) == 0 #allow overlapping cubes if they connect to different outputs

    def intersection_sizes(self, other: "CubePairMulti") -> np.ndarray:
        vcube_arr = mask_to_bool_array(self.vcube.cover_mask, self.vcube.n)
        other_vcube_arr = mask_to_bool_array(other.vcube.cover_mask, other.vcube.n)
        return np.bitwise_and(vcube_arr, other_vcube_arr) * (self.ccube.cover_mask & other.ccube.cover_mask).bit_count()

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

#TODO: Why doesn't this work
def update_SEMs(SEMs: list[np.ndarray], cube: CubePairMulti, nv: int, nc: int, m: int) -> list[np.ndarray]:
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

def get_problem_matrix_from_DV(V: np.ndarray) -> np.ndarray:
    nv = clog2(V.shape[0])
    m = clog2(V.shape[1])

    #Convert into a matrix of p (marginal) vectors
    Q = get_Q(m, lsb='right')
    P = np.empty((2 ** nv, 2 ** m))
    for i in range(2 ** nv):
        P[i, :] = Q @ V[i, :]

    return P[:, 1:]

def get_row_MVs_from_SEMs_efficient(Ks: list[np.ndarray]) -> np.ndarray:
    m = len(Ks)
    nv2 = Ks[0].shape[0]
    nc2 = Ks[0].shape[1]
    P = np.empty((nv2, 2 ** m))

    for i in range(2 ** m):
        K_anded = np.ones((nv2, nc2), dtype=np.bool_)
        mask = 1
        for j in range(m):
            if mask & i:
                K_anded = np.bitwise_and(K_anded, Ks[j])
            mask <<= 1
        P[:, i] = np.sum(K_anded, axis=1)
    return P

class Node:
    def __init__(self, pm: np.ndarray, cube_set: list[CubePairMulti], nv: int, nc: int, m: int):
        self.nv = nv
        self.nc = nc
        self.m = m
        self.initial_pm = pm
        self.pm = pm
        self.cube_set = cube_set
        #self.Ks = convert_cube_pairs_to_SEMs(cube_set, nv, nc, m)
        self.lit_count = sum([cs.lit_count for cs in cube_set]) if cube_set != [] else 0

    def overlaps(self, other: CubePairMulti) -> bool:
        if self.cube_set == []:
            return False
        for pair in self.cube_set:
            if pair.overlaps(other):
                return True
        return False

    def add_cube(self, cube: CubePairMulti) -> int: #returns a score value for the new cube
        new_cube_set = self.cube_set + [cube, ]

        #FIXME: this is incredibly slow, need a better way to do it!
        new_Ks = convert_cube_pairs_to_SEMs(new_cube_set, self.nv, self.nc, self.m)
        ps_covered = get_row_MVs_from_SEMs_efficient(new_Ks)[:, 1:]
        new_pm = self.initial_pm - ps_covered

        if np.any(new_pm < 0):
            return -1

        delta_pm = self.pm - new_pm
        #assert np.all(delta_pm_cool == delta_pm)
        score = np.sum(delta_pm)

        self.pm = new_pm
        self.cube_set = new_cube_set
        #self.Ks = new_Ks
        self.lit_count = sum([cs.lit_count for cs in self.cube_set]) if self.cube_set != [] else 0

        return score

    def is_solved(self) -> bool:
        return bool(np.all(self.pm == 0))

class NodePriorityQueue: #implemented as a max heap
    def __init__(self):
        self._heap = []
        self._counter = count()

    def push(self, node: Node, score: int) -> None:
        # -score makes heapq behave like a max heap.
        # counter prevents comparison between Node objects on score ties.
        heapq.heappush(self._heap, (-score, next(self._counter), node))

    def pop(self) -> Node:
        _, _, node = heapq.heappop(self._heap)
        return node

    def peek(self) -> Node:
        _, _, node = self._heap[0]
        return node

    def __len__(self) -> int:
        return len(self._heap)

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

def find_new_nodes(node: Node, vcubes: list[Cube], ccubes_by_size: dict) -> list[Node]:
    """Given a weight vector, find all of the valid cubes
    that do not intersect with an existing set of cubes. Order these by size

    The assumption is that the weight vector has already been updated to reflect the impact of the existing cube set
    """

    pq = NodePriorityQueue()
    for output_mask in count_in_bitcount_order(node.m):
        w = node.pm[:, output_mask - 1]
        if np.all(w == 0):
            continue
        max_w = np.max(w)
        max_multiplicity = 1 << (np.floor(np.log2(max_w)).astype(int))

        vcubes_under_consideration = copy.deepcopy(vcubes)

        current_multiplicity = 1
        while current_multiplicity <= max_multiplicity:
            proj = bool_array_to_mask(w >= current_multiplicity) #These are all the weights that have at least this multiplicity
            new_vcubes = []
            for vcube in vcubes_under_consideration:
                if vcube.covers(proj):
                    new_vcubes.append(vcube)
                    for ccube in ccubes_by_size[current_multiplicity]:
                        cube_pair = CubePairMulti(vcube, ccube, output_mask)

                        if node.overlaps(cube_pair):
                            continue

                        new_node = copy.copy(node)
                        score = new_node.add_cube(cube_pair)

                        if score == -1: #cube is not valid
                            continue

                        pq.push(new_node, score)
                            
            vcubes_under_consideration = new_vcubes
            current_multiplicity <<= 1

    return [node for _, _, node in pq._heap]

#This is the part of the code that corresponds directly to [Qian, 2017]
def branch_and_bound_opt_multi_output_v2(V: np.ndarray, iter_limit: int = 400000) -> list[np.ndarray]:
    nv = clog2(V.shape[0])
    nc = clog2(np.sum(V[0, :]))
    m = clog2(V.shape[1])

    initial_pm = get_problem_matrix_from_DV(V)

    N = Node(initial_pm, [], nv, nc, m)
    N_best = Node(initial_pm, [], nv, nc, m)
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
    timer = time.time()
    while stk != []:
        if iter_ctr % 1000 == 0:
            print(f"Iteration: {iter_ctr}, stack size: {len(stk)}")
            print(f"Time: {time.time() - timer}")
            timer = time.time()
        if iter_ctr > iter_limit:
            break
        iter_ctr += 1
        N = stk.pop()
        new_nodes = find_new_nodes(N, vcubes, ccubes_by_size)

        #print(f"Iteration problem vector: {N.pm}")

        while len(new_nodes) > 0:
            N_new = new_nodes.pop()
            if N_new.lit_count < n0:
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
    return Ks
