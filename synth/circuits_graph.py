import numpy as np
from collections import Counter
from sim.PTM import B_mat
from sim.Util import int_array

class Node():
    def __init__(self, idx) -> None:
        self.idx = idx
        self.src_node_idxs: list[int] = []
        self.dest_node_idxs: list[int] = []

        #used for topological sorting
        self.in_degree = 0

        self.primary_deps: set[int] = set()

class Edge():
    def __init__(self, src: int, dest: int, inv: bool = False):
        self.src = src
        self.dest = dest
        self.inv = inv

class AIG():
    #Node index layout: [0, num_inputs) are primary inputs, the next num_nodes are 2-input AND
    #gates, and the last num_outputs are primary outputs
    def __init__(self, num_inputs: int, num_outputs: int, num_nodes: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_nodes = num_nodes
        self.nodes: list[Node] = [Node(x) for x in range(num_inputs + num_outputs + num_nodes)]
        self.edges: list[Edge] = []

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        dest_node = self.nodes[edge.dest]
        src_node = self.nodes[edge.src]
        dest_node.src_node_idxs.append(edge.src)
        src_node.dest_node_idxs.append(edge.dest)

    def input_idxs(self) -> list[int]:
        return list(range(self.num_inputs))

    def and_idxs(self) -> list[int]:
        return list(range(self.num_inputs, self.num_inputs + self.num_nodes))

    def output_idxs(self) -> list[int]:
        start = self.num_inputs + self.num_nodes
        return list(range(start, start + self.num_outputs))

    def get_incoming_edges(self) -> list[list[Edge]]:
        """For each node, the list of edges entering it (in the order they were added)"""
        incoming: list[list[Edge]] = [[] for _ in self.nodes]
        for edge in self.edges:
            incoming[edge.dest].append(edge)
        return incoming

    def topological_sort(self) -> list[Node]:
        queue: list[Node] = []
        for node in self.nodes:
            node.in_degree = len(node.src_node_idxs)
            if node.in_degree == 0:
                queue.append(node)

        top_sort: list[Node] = []
        while len(queue) > 0:
            node = queue.pop(0)
            top_sort.append(node)
            for dest_node_idx in node.dest_node_idxs:
                dest_node = self.nodes[dest_node_idx]
                dest_node.in_degree -= 1
                if dest_node.in_degree == 0:
                    queue.append(dest_node)

        return top_sort

    def get_input_deps(self):
        #For each node, figure out which primary inputs it depends on, store as a set of integers
        supports = node_supports(self)
        for node in self.nodes:
            if len(node.src_node_idxs) > 2:
                raise ValueError(f"Node {node.idx} has {len(node.src_node_idxs)} source nodes, which is not supported")
            node.primary_deps.update(supports[node.idx])

        #detect reconvergent fanout - this is where the two fanin cones of a node overlap
        for idx, shared in reconvergent_nodes(self).items():
            print(f"Node {idx} has reconvergent fanout: {shared}")

def build_MUX_AIG() -> AIG:
    mux_aig = AIG(3, 1, 3)

    #Input nodes 0, 1, 2 to AND gates nodes 3 and 4
    mux_aig.add_edge(Edge(0, 3))
    mux_aig.add_edge(Edge(1, 4))
    mux_aig.add_edge(Edge(2, 3))
    mux_aig.add_edge(Edge(2, 4, True))

    #AND gate nodes 3 and 4 to OR gate node 5
    mux_aig.add_edge(Edge(3, 5, True))
    mux_aig.add_edge(Edge(4, 5, True))

    #OR gate node 5 to output node 6
    mux_aig.add_edge(Edge(5, 6, True))

    return mux_aig

def build_notes_example1_AIG() -> AIG:
    aig = AIG(7, 2, 7)
    aig.add_edge(Edge(1, 7))
    aig.add_edge(Edge(2, 7))
    aig.add_edge(Edge(7, 10))
    aig.add_edge(Edge(3, 8))
    aig.add_edge(Edge(4, 8))
    aig.add_edge(Edge(8, 10))
    aig.add_edge(Edge(8, 11))
    aig.add_edge(Edge(5, 9))
    aig.add_edge(Edge(6, 9))
    aig.add_edge(Edge(9, 11))
    aig.add_edge(Edge(10, 13))
    aig.add_edge(Edge(11, 13))
    aig.add_edge(Edge(13, 15))
    aig.add_edge(Edge(10, 12))
    aig.add_edge(Edge(0, 12))
    aig.add_edge(Edge(12, 14))

    return aig

def build_notes_example2_AIG() -> AIG:
    aig = AIG(4, 5, 1)
    aig.add_edge(Edge(0, 6))
    aig.add_edge(Edge(6, 8))
    aig.add_edge(Edge(8, 9))
    aig.add_edge(Edge(1, 4))
    aig.add_edge(Edge(4, 6))
    aig.add_edge(Edge(4, 7))
    aig.add_edge(Edge(7, 8))
    aig.add_edge(Edge(2, 4))
    aig.add_edge(Edge(2, 5))
    aig.add_edge(Edge(3, 5))
    aig.add_edge(Edge(5, 7))

    return aig

#PTM --> AIG

#Literals are AIGER-style: lit = 2 * node_idx + inv. AIGs have no constant node, so constant
#functions are carried around as these sentinels until they reach a primary output
FALSE_LIT = -2
TRUE_LIT = -1

class AIGBuilder():
    """Incrementally build an AIG out of literals. Node indices aren't known until the whole
    circuit has been described, so the AND gates are buffered and materialized by build()"""
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self.and_gates: list[tuple[int, int]] = [] #(fanin literal, fanin literal) per AND gate
        self.outputs: list[int] = []
        self.cache: dict[tuple[int, int], int] = {} #structural hashing, shares gates across outputs
        self.const_false = None

    def input_lit(self, idx: int, inv: bool = False) -> int:
        if idx >= self.num_inputs:
            raise ValueError(f"Input {idx} is out of range for {self.num_inputs} inputs")
        return 2 * idx + int(inv)

    def inv_lit(self, lit: int) -> int:
        if lit == FALSE_LIT:
            return TRUE_LIT
        if lit == TRUE_LIT:
            return FALSE_LIT
        return lit ^ 1

    def and_lit(self, a: int, b: int) -> int:
        if a == FALSE_LIT or b == FALSE_LIT:
            return FALSE_LIT
        if a == TRUE_LIT:
            return b
        if b == TRUE_LIT:
            return a
        if a == b:
            return a
        if a == b ^ 1:
            return FALSE_LIT

        key = (min(a, b), max(a, b))
        if key in self.cache:
            return self.cache[key]
        lit = 2 * (self.num_inputs + len(self.and_gates))
        self.and_gates.append(key)
        self.cache[key] = lit
        return lit

    def or_lit(self, a: int, b: int) -> int:
        return self.inv_lit(self.and_lit(self.inv_lit(a), self.inv_lit(b)))

    def _reduce_balanced(self, lits: list[int], op) -> int:
        #Combine pairwise rather than linearly, to keep the AIG depth logarithmic
        lits = list(lits)
        while len(lits) > 1:
            next_lits = [op(lits[i], lits[i+1]) for i in range(0, len(lits) - 1, 2)]
            if len(lits) % 2 == 1:
                next_lits.append(lits[-1])
            lits = next_lits
        return lits[0]

    def and_all(self, lits: list[int]) -> int:
        if len(lits) == 0: #empty product term (the cube with every input don't-care)
            return TRUE_LIT
        return self._reduce_balanced(lits, self.and_lit)

    def or_all(self, lits: list[int]) -> int:
        if len(lits) == 0: #empty sum (the function is constant 0)
            return FALSE_LIT
        return self._reduce_balanced(lits, self.or_lit)

    def add_output(self, lit: int):
        self.outputs.append(lit)

    def _get_const_false(self) -> int:
        """Constant 0 as x0 AND NOT x0 - the only way to express a constant without a const node"""
        if self.const_false is None:
            if self.num_inputs == 0:
                raise ValueError("Cannot build a constant function in an AIG with no inputs")
            x0 = self.input_lit(0)
            self.const_false = 2 * (self.num_inputs + len(self.and_gates))
            self.and_gates.append((x0, x0 ^ 1)) #appended directly, and_lit would simplify it away
        return self.const_false

    def build(self) -> AIG:
        outputs = []
        for lit in self.outputs:
            if lit == FALSE_LIT:
                lit = self._get_const_false()
            elif lit == TRUE_LIT:
                lit = self.inv_lit(self._get_const_false())
            outputs.append(lit)

        aig = AIG(self.num_inputs, len(outputs), len(self.and_gates))
        for gate_idx, (a, b) in enumerate(self.and_gates):
            dest = self.num_inputs + gate_idx
            aig.add_edge(Edge(a >> 1, dest, bool(a & 1)))
            aig.add_edge(Edge(b >> 1, dest, bool(b & 1)))

        out_start = self.num_inputs + len(self.and_gates)
        for out_idx, lit in enumerate(outputs):
            aig.add_edge(Edge(lit >> 1, out_start + out_idx, bool(lit & 1)))
        return aig

def ptm_to_truth_table(Mf, lsb='right'):
    """Convert a PTM into the (2 ** n, k) array of output bits it encodes"""
    n, k = np.log2(np.array(Mf.shape)).astype(np.int32)
    return np.array(Mf @ B_mat(k, lsb=lsb), dtype=np.bool_)

#A cube is a pair (bits, dashes) of bitmasks over the 2 ** n row index, where a set bit of
#dashes marks a don't-care position and the corresponding bit of bits is held at 0
def prime_implicants(onset: list[int], n: int) -> list[tuple[int, int]]:
    """Quine-McCluskey: repeatedly merge cubes that differ in a single non-dashed position"""
    cubes = {(m, 0) for m in onset}
    primes = []
    while len(cubes) > 0:
        merged = set()
        next_cubes = set()
        for bits, dashes in cubes:
            for i in range(n):
                bit = 1 << i
                if dashes & bit:
                    continue
                neighbor = (bits ^ bit, dashes)
                if neighbor in cubes:
                    merged.add((bits, dashes))
                    merged.add(neighbor)
                    next_cubes.add((bits & ~bit, dashes | bit))
        primes += sorted(cubes - merged)
        cubes = next_cubes
    return primes

def sop_cover(onset: list[int], n: int, minimize=True) -> list[tuple[int, int]]:
    """Cover the onset with cubes. Exact minimization is NP-hard, so take the essential prime
    implicants and then greedily add the prime that covers the most of what's left"""
    if len(onset) == 0:
        return []
    if not minimize:
        return [(m, 0) for m in onset]

    primes = prime_implicants(onset, n)
    covers = {p: {m for m in onset if m & ~p[1] == p[0]} for p in primes}

    chosen = []
    uncovered = set(onset)
    for m in onset: #essential primes - minterms covered by exactly one prime implicant
        containing = [p for p in primes if m in covers[p]]
        if len(containing) == 1 and containing[0] not in chosen:
            chosen.append(containing[0])
            uncovered -= covers[containing[0]]

    while len(uncovered) > 0:
        best = max(primes, key=lambda p: (len(covers[p] & uncovered), -bin(p[1]).count('1')))
        chosen.append(best)
        uncovered -= covers[best]
    return chosen

def cube_literals(builder: AIGBuilder, cube: tuple[int, int], n: int, lsb='right') -> list[int]:
    """The input literals of one product term, ordered by input index"""
    bits, dashes = cube
    lits = []
    for i in range(n):
        if dashes & (1 << i):
            continue
        #Row index i of the PTM is B_mat(n, lsb)[i], so for lsb='right' input 0 is the MSB
        input_idx = n - 1 - i if lsb == 'right' else i
        lits.append(builder.input_lit(input_idx, inv=not bits & (1 << i)))
    return sorted(lits)

def ptm_to_aig(Mf, lsb='right', minimize=True) -> AIG:
    """Synthesize an AIG from a PTM. Each output is covered by a sum of products, which maps
    onto AND gates with inverted edges (an OR is an AND with all three of its edges inverted).
    Product terms shared between outputs are merged by the builder's structural hashing"""
    tt = ptm_to_truth_table(Mf, lsb=lsb)
    n = np.log2(tt.shape[0]).astype(np.int32)
    k = tt.shape[1]

    builder = AIGBuilder(n)
    for j in range(k):
        onset = np.flatnonzero(tt[:, j]).tolist()
        cubes = sop_cover(onset, n, minimize=minimize)
        products = [builder.and_all(cube_literals(builder, cube, n, lsb=lsb)) for cube in cubes]
        builder.add_output(builder.or_all(products))
    return builder.build()

def circ_to_aig(circ, lsb='right', minimize=True) -> AIG:
    """Synthesize an AIG from a combinational Circ (sim.circs.circs)"""
    return ptm_to_aig(circ.get_PTM(lsb=lsb), lsb=lsb, minimize=minimize)

#AIG --> PTM

def aig_eval_nodes(aig: AIG, bs_mat) -> list:
    """Evaluate every node of the AIG on a (num_inputs, N) matrix of input bit vectors"""
    bs_mat = np.array(bs_mat, dtype=np.bool_)
    if bs_mat.shape[0] != aig.num_inputs:
        raise ValueError(f"Expected {aig.num_inputs} input rows, got {bs_mat.shape[0]}")

    incoming = aig.get_incoming_edges()
    vals = [None for _ in aig.nodes]
    for node in aig.topological_sort():
        edges = incoming[node.idx]
        if len(edges) == 0:
            if node.idx >= aig.num_inputs:
                raise ValueError(f"Node {node.idx} is not a primary input but has no fanin")
            vals[node.idx] = bs_mat[node.idx, :]
        else:
            #An inverted edge complements the signal travelling along it
            val = np.bitwise_xor(vals[edges[0].src], edges[0].inv)
            for edge in edges[1:]:
                val = np.bitwise_and(val, np.bitwise_xor(vals[edge.src], edge.inv))
            vals[node.idx] = val
    return vals

def aig_eval(aig: AIG, bs_mat):
    """Evaluate the AIG's primary outputs on a (num_inputs, N) matrix of input bit vectors"""
    vals = aig_eval_nodes(aig, bs_mat)
    return np.array([vals[idx] for idx in aig.output_idxs()])

def aig_to_ptm(aig: AIG, lsb='right'):
    """The PTM of the whole AIG, matching the convention of CombCirc.get_PTM"""
    Bn = B_mat(aig.num_inputs, lsb=lsb)
    outs = aig_eval(aig, Bn.T)
    out_ints = int_array(outs.T, lsb=lsb)

    Mf = np.zeros((2 ** aig.num_inputs, 2 ** aig.num_outputs), dtype=np.bool_)
    Mf[np.arange(2 ** aig.num_inputs), out_ints] = True
    return Mf

def aig_logic_cone(aig: AIG, node_idx: int) -> tuple[list[int], list[int]]:
    """The transitive fanin of a node, as (cone node indices, primary input indices)"""
    incoming = aig.get_incoming_edges()
    cone, queue = set(), [node_idx]
    while len(queue) > 0:
        idx = queue.pop(0)
        if idx in cone:
            continue
        cone.add(idx)
        queue += [edge.src for edge in incoming[idx]]
    return sorted(cone), sorted(cone & set(aig.input_idxs()))

def aig_cone_ptm(aig: AIG, node_idx: int, lsb='right'):
    """The single-output PTM of the logic cone feeding one node, over only the primary inputs
    that node depends on. Returns (PTM, those input indices)"""
    _, deps = aig_logic_cone(aig, node_idx)
    Bd = B_mat(len(deps), lsb=lsb)

    bs_mat = np.zeros((aig.num_inputs, 2 ** len(deps)), dtype=np.bool_)
    bs_mat[deps, :] = Bd.T
    out = aig_eval_nodes(aig, bs_mat)[node_idx]

    Mf = np.zeros((2 ** len(deps), 2), dtype=np.bool_)
    Mf[np.arange(2 ** len(deps)), 1 * out] = True
    return Mf, deps

def aig_node_ptms(aig: AIG, lsb='right') -> dict:
    """The cone PTM of every AND gate and primary output, keyed by node index"""
    return {idx: aig_cone_ptm(aig, idx, lsb=lsb) for idx in aig.and_idxs() + aig.output_idxs()}

#AIG --> supergate graph

#A supergate is the smallest subcircuit rooted at a node whose boundary inputs have pairwise
#disjoint support (Seth/Pan/Agrawal, PREDICT). Every reconvergent path is then contained inside
#some supergate, and the signals passed between supergates are mutually independent
class Supergate():
    def __init__(self, root: int, interior: list[int], inputs: list[int]):
        self.root = root          #the node whose value this supergate computes
        self.interior = interior  #nodes evaluated inside the supergate, including the root
        self.inputs = inputs      #boundary signals, mutually independent by construction

    def __repr__(self):
        return f"Supergate(root={self.root}, interior={self.interior}, inputs={self.inputs})"

def node_supports(aig: AIG) -> list[set[int]]:
    """For each node, the set of primary inputs in its transitive fanin"""
    incoming = aig.get_incoming_edges()
    supports = [set() for _ in aig.nodes]
    for node in aig.topological_sort():
        if len(incoming[node.idx]) == 0:
            supports[node.idx].add(node.idx)
        else:
            for edge in incoming[node.idx]:
                supports[node.idx] |= supports[edge.src]
    return supports

def shared_support(idxs, supports: list[set[int]]) -> set[int]:
    """The primary inputs that more than one of the given nodes depends on"""
    counts = Counter()
    for idx in idxs:
        counts.update(supports[idx])
    return {pi for pi, cnt in counts.items() if cnt > 1}

def reconvergent_nodes(aig: AIG) -> dict:
    """The nodes whose fanin cones overlap, mapped to the primary inputs they share"""
    supports = node_supports(aig)
    incoming = aig.get_incoming_edges()
    found = {}
    for node in aig.nodes:
        shared = shared_support([edge.src for edge in incoming[node.idx]], supports)
        if len(shared) > 0:
            found[node.idx] = shared
    return found

def supergate(aig: AIG, root: int, supports=None, incoming=None) -> Supergate:
    """Grow the fanin cone of `root` until its boundary inputs have pairwise disjoint support.
    Expanding every conflicted input is what minimality requires: if two inputs share a primary
    input, that primary input has to end up on the boundary itself, which means everything
    between it and the boundary must be absorbed"""
    if supports is None:
        supports = node_supports(aig)
    if incoming is None:
        incoming = aig.get_incoming_edges()

    interior = {root}
    inputs = {edge.src for edge in incoming[root]}
    while True:
        shared = shared_support(inputs, supports)
        #A conflicted input that is itself a primary input is the shared source, so leave it be
        expandable = [idx for idx in inputs if supports[idx] & shared and len(incoming[idx]) > 0]
        if len(expandable) == 0:
            break
        for idx in expandable:
            interior.add(idx)
            inputs.remove(idx)
            inputs |= {edge.src for edge in incoming[idx]}
        inputs -= interior #an absorbed node stays absorbed, even if it feeds the new boundary
    return Supergate(root, sorted(interior), sorted(inputs))

class SupergateGraph():
    def __init__(self, aig: AIG, supergates: dict):
        self.aig = aig
        self.supergates: dict[int, Supergate] = supergates #keyed by root node index

    def roots(self) -> list[int]:
        """Supergate roots, ordered so that a supergate comes after everything feeding it"""
        order = {node.idx: i for i, node in enumerate(self.aig.topological_sort())}
        return sorted(self.supergates.keys(), key=lambda idx: order[idx])

    def fanin_roots(self, root: int) -> list[int]:
        """The supergates feeding this one"""
        return [idx for idx in self.supergates[root].inputs if idx in self.supergates]

    def primary_inputs(self, root: int) -> list[int]:
        """The inputs of this supergate that come straight from primary inputs"""
        return [idx for idx in self.supergates[root].inputs if idx not in self.supergates]

    def covered_nodes(self) -> set[int]:
        covered = set()
        for sg in self.supergates.values():
            covered |= set(sg.interior)
        return covered

    def duplicated_nodes(self) -> set[int]:
        """Nodes appearing in more than one supergate. Cutting at these would break the
        independence property, so the duplication is inherent to the decomposition"""
        counts = Counter()
        for sg in self.supergates.values():
            counts.update(sg.interior)
        return {idx for idx, cnt in counts.items() if cnt > 1}

    def verify(self) -> bool:
        """Check that no reconvergence crosses a supergate boundary"""
        supports = node_supports(self.aig)
        for sg in self.supergates.values():
            if len(shared_support(sg.inputs, supports)) > 0:
                return False
        return True

    def print_summary(self):
        widths = [len(sg.inputs) for sg in self.supergates.values()]
        sizes = [len(sg.interior) for sg in self.supergates.values()]
        print(f"{len(self.supergates)} supergates, largest has {max(sizes)} nodes "
              f"and {max(widths)} inputs (PTM rows: {2 ** max(widths)})")
        print(f"duplicated nodes: {sorted(self.duplicated_nodes())}")
        for root in self.roots():
            sg = self.supergates[root]
            print(f"  {sg} <- supergates {self.fanin_roots(root)}, inputs {self.primary_inputs(root)}")

def supergate_decomposition(aig: AIG, roots=None) -> SupergateGraph:
    """Decompose the AIG into supergates, starting from the primary outputs and recursing on
    each boundary input until everything bottoms out at the primary inputs"""
    supports = node_supports(aig)
    incoming = aig.get_incoming_edges()

    supergates = {}
    queue = list(aig.output_idxs() if roots is None else roots)
    while len(queue) > 0:
        root = queue.pop(0)
        if root in supergates or len(incoming[root]) == 0: #already done, or a primary input
            continue
        sg = supergate(aig, root, supports, incoming)
        supergates[root] = sg
        queue += sg.inputs
    return SupergateGraph(aig, supergates)

def supergate_eval(aig: AIG, sg: Supergate, in_vals, incoming=None):
    """Evaluate a supergate's root given a (len(sg.inputs), N) matrix of boundary input values"""
    if incoming is None:
        incoming = aig.get_incoming_edges()
    in_vals = np.array(in_vals, dtype=np.bool_)
    if in_vals.shape[0] != len(sg.inputs):
        raise ValueError(f"Expected {len(sg.inputs)} input rows, got {in_vals.shape[0]}")

    vals = {idx: in_vals[i, :] for i, idx in enumerate(sg.inputs)}
    interior = set(sg.interior)
    for node in aig.topological_sort(): #the interior in topological order
        if node.idx not in interior:
            continue
        edges = incoming[node.idx]
        val = np.bitwise_xor(vals[edges[0].src], edges[0].inv)
        for edge in edges[1:]:
            val = np.bitwise_and(val, np.bitwise_xor(vals[edge.src], edge.inv))
        vals[node.idx] = val
    return vals[sg.root]

def supergate_ptm(aig: AIG, sg: Supergate, lsb='right', incoming=None):
    """The single-output PTM of one supergate, over its boundary inputs rather than over the
    primary inputs. Those inputs are independent, so the input PTV of this matrix is just the
    Kronecker product of the per-signal PTVs"""
    n = len(sg.inputs)
    out = supergate_eval(aig, sg, B_mat(n, lsb=lsb).T, incoming=incoming)

    Mf = np.zeros((2 ** n, 2), dtype=np.bool_)
    Mf[np.arange(2 ** n), 1 * out] = True
    return Mf

def supergate_ptms(sg_graph: SupergateGraph, lsb='right') -> dict:
    """The PTM of every supergate, keyed by root node index"""
    incoming = sg_graph.aig.get_incoming_edges()
    return {root: supergate_ptm(sg_graph.aig, sg, lsb=lsb, incoming=incoming)
            for root, sg in sg_graph.supergates.items()}

def test_topological_sort():
    example_aig = build_notes_example2_AIG()
    example_aig.get_input_deps()
    pass

def test_ptm_aig_roundtrip():
    from sim.circs.circs import C_MUX_ADD, C_MAC, C_RCED, C_AND_OR, C_MUX_PAIR, C_XOR

    for circ in [C_XOR(), C_MUX_ADD(), C_MAC(), C_RCED(), C_AND_OR(), C_MUX_PAIR()]:
        Mf = circ.get_PTM()
        aig = circ_to_aig(circ)
        assert np.array_equal(aig_to_ptm(aig), Mf)
        print(f"{circ.name}: n={aig.num_inputs} k={aig.num_outputs} AND gates={aig.num_nodes}")

    #The hand-built MUX AIG should survive a trip through its own PTM
    mux_aig = build_MUX_AIG()
    Mf = aig_to_ptm(mux_aig)
    assert np.array_equal(aig_to_ptm(ptm_to_aig(Mf)), Mf)

def test_supergate_decomposition():
    from sim.circs.circs import C_RCED

    for aig in [build_MUX_AIG(), build_notes_example1_AIG(), circ_to_aig(C_RCED())]:
        sgd = supergate_decomposition(aig)
        assert sgd.verify() #no reconvergence crosses a supergate boundary

        #Every reconvergent node must have landed inside some supergate
        covered = sgd.covered_nodes()
        assert all(idx in covered for idx in reconvergent_nodes(aig))

        #Composing the supergates must reproduce the AIG's own evaluation
        bs_mat = B_mat(aig.num_inputs).T
        vals = {idx: bs_mat[idx, :] for idx in aig.input_idxs()}
        for root in sgd.roots():
            sg = sgd.supergates[root]
            vals[root] = supergate_eval(aig, sg, np.array([vals[i] for i in sg.inputs]))
        outs = np.array([vals[idx] for idx in aig.output_idxs()])
        assert np.array_equal(outs, aig_eval(aig, bs_mat))

        sgd.print_summary()
        print()