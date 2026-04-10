from abc import abstractmethod
import numpy as np
import sympy as sp
from functools import reduce
from sim.ReSC import ReSC, B_GAMMA
from synth.sat import C_to_cgroups_and_sign
from sim.PTM import TT_to_ptm
from sim.Util import bit_vec_arr_to_int, clog2
from sim.PTM import B_mat
from sim.SCC import scc

class Circ:
    def __init__(self, n, m, nc=0, Cin=None, name="Circ"):
        self.n = n #total number of inputs (including constant inputs)
        self.m = m #total number of outputs
        self.nc = nc #number of 0.5-valued constant inputs
        self.nv = self.n - self.nc
        if Cin is not None:
            self.cgroups, self.signs = C_to_cgroups_and_sign(Cin)
        else:
            # Default to uncorrelated inputs (identity correlation matrix)
            self.cgroups = list(range(self.nv))
            self.signs = [1] * self.nv
        self.nv_star = len(np.unique(self.cgroups))
        self.name = name

    def parr_mod(self, parr):
        """Used to handle non-0.5 constant inputs"""
        return parr
    
    def get_rns_width(self, w):
        return w * self.nv_star + self.nc

    def get_Nmax(self, w):
        return 2 ** self.get_rns_width(w)
    
    def lzd_func(self, k):
        return k
    
    @abstractmethod
    def run(self, bs_mat):
        pass
    
    @abstractmethod
    def correct(self, parr):
        pass

    @abstractmethod
    def get_PTM(self, lsb='right'):
        pass

class CombCirc(Circ):
    def __init__(self, n, m, nc=0, Cin=None, name="CombCirc"):
        super().__init__(n, m, nc, Cin, name)
    
    def get_PTM(self, lsb='right'):
        Bn = B_mat(self.n, lsb=lsb) #When printed out, the rightmost columns are the constant inputs
        output = self.run(Bn.T)

        if len(output.shape) == 1:
            output = np.expand_dims(output, axis=0)
        
        output_ints = bit_vec_arr_to_int(output, lsb=lsb)

        #TODO: Convert to a sparse representation
        Mf = np.zeros((2 ** self.n, 2 ** self.m), dtype=bool)
        for i in range(2 ** self.n):
            Mf[i, output_ints[i]] = True
        return Mf

class SeqCirc(Circ):
    def __init__(self, n, m, nc, ns, Cin=None, name="SeqCirc"):
        self.ns = ns #number of states in the non-extended FSM
        super().__init__(n, m, nc, Cin, name)

    def get_vars(self):
        return sp.symbols(" ".join([f"x{i}" for i in range(self.n)]))

    @abstractmethod
    #Transition list in the form of (state_index_src, state_index_dest, func)
    def get_transition_list(self):
        pass

    @abstractmethod
    #return a matrix of size (2**n, m, ns)
    def get_mealy_TTs(self):
        pass
    
    #Get the mealy PTM for each of the individual states of the FSM
    def get_PTM(self, pi, lsb='right'):
        #Create the weighted sum of the individual state PTMs
        #NOTE: This may have some relevance to how we'd extend COOPT to sequential elements
        # We can run COOPT individually on all of the state PTMs, 
        # and the result should be SE to the original FSM
        TTs = self.get_mealy_TTs()
        overall_ptm = sp.ZeroMatrix(2 ** self.n, 2 ** self.m)
        for i, pii in enumerate(pi):
            TT = sp.Matrix(1 * TT_to_ptm(TTs[:, :, i], self.n, self.m, lsb=lsb))
            overall_ptm += pii * TT
        return overall_ptm

#A circuit directly defined by its PTM instead of by its functional behavior
class PTM_Circ(Circ):
    def __init__(self, Mf, c: Circ):
        self.Mf = Mf
        self.circ = c
        super().__init__(c.n, c.m, c.nc, c.cgroups, c.name)

    def run(self, bs_mat):
        from sim.PTM import apply_ptm_to_bs
        return apply_ptm_to_bs(bs_mat, self.Mf, lsb='right')

    def correct(self, parr):
        return self.circ.correct(parr)

    def get_PTM(self, lsb='right'):
        return self.Mf

class C_WIRE(CombCirc):
    def __init__(self, n, Cin=None):
        super().__init__(n, n, 0, Cin, "WIRE")

    def run(self, bs_mat):
        return bs_mat
    
    def correct(self, parr):
        return parr

class C_AND_N(CombCirc):
    def __init__(self, n, Cin=None):
        super().__init__(n, 1, 0, Cin, "AND Gate n={}".format(n))

    def run(self, bs_mat):
        _, N = bs_mat.shape
        bs_out = np.ones((N,), dtype=np.bool_)
        for j in range(self.n):
            bs_out = np.bitwise_and(bs_out, bs_mat[j, :])
        return bs_out

    def correct(self, parr):
        return reduce(lambda x, y: x*y, parr)
    
    def lzd_func(self, k):
        return k ** self.n
    
class C_OR_N(CombCirc):
    def __init__(self, n, Cin=None):
        super().__init__(n, 1, 0, Cin, "OR Gate n={}".format(n))

    def run(self, bs_mat):
        _, N = bs_mat.shape
        bs_out = np.zeros((N,), dtype=np.bool_)
        for j in range(self.n):
            bs_out = np.bitwise_or(bs_out, bs_mat[j, :])
        return bs_out
    
    def correct(self, parr):
        return reduce(lambda x, y: x+y-x*y, parr)

class C_XOR(CombCirc):
    def __init__(self):
        super().__init__(2, 1, 0, [0, 1], "XOR Gate")

    def run(self, bs_mat):
        return np.bitwise_xor(bs_mat[0, :], bs_mat[1, :])
    
    def correct(self, parr):
        return parr[0] + parr[1] - 2*parr[0]*parr[1]
    
class C_MUX_ADD(CombCirc):
    def __init__(self, corr_inputs=True):
        if corr_inputs:
            cgroups = [0, 0]
        else:
            cgroups = [0, 1]
        super().__init__(3, 1, 1, cgroups, "MUX Gate")

    def run(self, bs_mat):
        return mux(bs_mat[0, :], bs_mat[1, :], bs_mat[2, :])

    def correct(self, parr):
        return 0.5 * (parr[0] + parr[1])
    
class C_MAC(CombCirc):
    def __init__(self):
        super().__init__(5, 1, 1, [0, 0, 1, 1], "MAC")
    
    def run(self, bs_mat):
        a1 = np.bitwise_and(bs_mat[0, :], bs_mat[2, :])
        a2 = np.bitwise_and(bs_mat[1, :], bs_mat[3, :])
        return mux(a1, a2, bs_mat[4, :])

    def correct(self, parr):
        return 0.5 * (parr[0] * parr[2] + parr[1] * parr[3])
    
    def lzd_func(self, k):
        return k ** 2
    
class C_MUX_PAIR(CombCirc):
    def __init__(self):
        super().__init__(5, 2, 1, [0, 0, 0, 0], "MUX_PAIR")

    def run(self, bs_mat):
        return np.array([
            mux(bs_mat[0, :], bs_mat[1, :], bs_mat[4, :]),
            mux(bs_mat[2, :], bs_mat[3, :], bs_mat[4, :])
        ])

    def correct(self, parr):
        return 0.5 * np.array([
            parr[0] + parr[1],
            parr[2] + parr[3]
        ])

class C_MAJ_PAIR(CombCirc):
    def __init__(self):
        super().__init__(5, 2, 1, [0, 0, 0, 0], "MAJ_PAIR")

    def run(self, bs_mat):
        return np.array([
            maj(bs_mat[0, :], bs_mat[1, :], bs_mat[4, :]),
            maj(bs_mat[2, :], bs_mat[3, :], bs_mat[4, :])
        ])

    def correct(self, parr):
        return 0.5 * np.array([
            parr[0] + parr[1],
            parr[2] + parr[3]
        ])
    
class C_RCED(CombCirc):
    def __init__(self):
        super().__init__(5, 1, 1, [0, 0, 0, 0], "RCED")

    def run(self, bs_mat):
        return robert_cross(*[bs_mat[x, :] for x in range(5)])

    def correct(self, parr):
        return 0.5 * (np.abs(parr[0] - parr[1]) + np.abs(parr[2] - parr[3]))

class C_MAX(CombCirc):
    def __init__(self):
        super().__init__(2, 1, 0, [0, 0], "MAX")

    def run(self, bs_mat):
        return np.bitwise_or(bs_mat[0, :], bs_mat[1, :])

    def correct(self, parr):
        return np.maximum(parr[0], parr[1])
    
class C_Gamma(CombCirc):
    def __init__(self):
        super().__init__(13, 1, 0, [0, 1, 2, 3, 4, 5] + [6 for _ in range(7)], "ReSC Gamma")
        #Not sure how many 0.5-valued constants are required to generate the inputs to the ReSC circuit

    def parr_mod(self, parr):
        return np.array([parr[0],] * 6 + B_GAMMA)

    def run(self, bs_mat):
        return ReSC(bs_mat).flatten()
    
    def correct(self, parr):
        return parr[0] ** 0.45

class C_MAC_N(CombCirc):
    def __init__(self, nX, bipolar=True, relu=False):
        self.bipolar = bipolar
        if relu and not bipolar:
            raise ValueError("ReLU is only supported with bipolar encoding")
        self.relu = relu
        self.depth = np.log2(nX).astype(int)
        if self.depth != clog2(nX):
            raise ValueError("C_MAC_N does not currently support nX that isn't a power of 2")
        self.nX = nX
        #Organization: [X_1, ..., X_nX, 0.5 (relu)], [W_1,...,W_nX], [c_1,...,c_log2(Nx)]
        #Interesting problem: Is it better to correlate the relu input with the data or weights?
        super().__init__(2 * nX + relu*1, 1, self.depth, [0 for _ in range(nX + relu*1)] + [1 for _ in range(nX)], "MAC_N")
    
    def run(self, bs_mat):
        N = bs_mat.shape[1]

        multed = np.empty((self.nX, N), dtype=np.bool_)
        ro = self.relu * 1 #relu offset alias
        for i in range(self.nX):
            if self.bipolar:
                multed[i, :] = np.bitwise_not(np.bitwise_xor(bs_mat[i, :], bs_mat[self.nX + i + ro, :]))
            else:
                multed[i, :] = np.bitwise_and(bs_mat[i, :], bs_mat[self.nX + i, :])
        
        mSz = int(self.nX / 2)
        nc_idx = 0
        mi_next = np.empty((mSz, N), dtype=np.bool_)
        mi = np.empty((mSz, N), dtype=np.bool_)
        for i in range(self.depth):
            for j in range(mSz):
                if nc_idx == 0:
                    mi_next[j] = mux(multed[2*j, :], multed[2*j+1, :], bs_mat[2 * self.nX + nc_idx + ro, :])
                else:
                    mi_next[j] = mux(mi[2*j, :], mi[2*j+1, :], bs_mat[2 * self.nX + nc_idx + ro, :])
            mSz = int(mSz / 2)
            mi = mi_next.copy()
            mi_next = np.empty((mSz, N), dtype=np.bool_)
            nc_idx += 1
        #print(scc(mi, bs_mat[self.nX, :])) #very poor correlation here
        if self.relu:
            mi = np.bitwise_or(mi, bs_mat[self.nX, :])
        return mi

    def correct(self, parr):
        """Assuming parr is in unipolar form, optionally convert to bipolar and back"""
        a = 0
        ro = self.relu * 1 #relu offset alias
        for i in range(self.nX):
            x = parr[i]
            w = parr[self.nX + i + ro]
            if self.bipolar:
                x = 1 - 2 * x
                w = 1 - 2 * w
            a += w * x
        r = a / self.nX
        if self.relu:
            r = max(0, r)
        if self.bipolar:
            return (r + 1) / 2
        return r

class C_SobelMuxes(CombCirc):
    def __init__(self, Cin=None, use_maj=False):
        self.use_maj = use_maj
        if Cin is None:
            Cin = [0] * 9
        super().__init__(12, 4, 3, Cin, "GBED")

    def run(self, bs_mat):
        if self.use_maj:
            func = maj
        else:
            func = mux
        p = [bs_mat[i, :] for i in range(9)]
        s1, s2, s3 = bs_mat[9, :], bs_mat[10, :], bs_mat[11, :]

        # Gx: Gaussian-weighted column sums, then XOR for |diff|
        # left  = (p00 + 2*p10 + p20) / 4
        # right = (p02 + 2*p12 + p22) / 4
        left  = func(func(p[0], p[6], s1), p[3], s2)
        right = func(func(p[2], p[8], s1), p[5], s2)

        # Gy: Gaussian-weighted row sums, then XOR for |diff|
        # top    = (p00 + 2*p01 + p02) / 4
        # bottom = (p20 + 2*p21 + p22) / 4
        top    = func(func(p[0], p[2], s1), p[1], s2)
        bottom = func(func(p[6], p[8], s1), p[7], s2)

        return np.array([left, right, top, bottom])

class C_Sobel(CombCirc):
    """See reference: Comparing the Robustness of Deterministic and Stochastic Edge Detection Circuits to Transmission Noise
    for Sobel filter design
    """
    def __init__(self, Cin=None, use_maj=False):
        self.use_maj = use_maj
        if Cin is None:
            Cin = [0] * 9
        super().__init__(12, 1, 3, Cin, "GBED")

    def run(self, bs_mat):
        if self.use_maj:
            func = maj
        else:
            func = mux
        p = [bs_mat[i, :] for i in range(9)]
        s1, s2, s3 = bs_mat[9, :], bs_mat[10, :], bs_mat[11, :]

        # Gx: Gaussian-weighted column sums, then XOR for |diff|
        # left  = (p00 + 2*p10 + p20) / 4
        # right = (p02 + 2*p12 + p22) / 4
        left  = func(func(p[0], p[6], s1), p[3], s2)
        right = func(func(p[2], p[8], s1), p[5], s2)
        gx = np.bitwise_xor(left, right)

        # Gy: Gaussian-weighted row sums, then XOR for |diff|
        # top    = (p00 + 2*p01 + p02) / 4
        # bottom = (p20 + 2*p21 + p22) / 4
        top    = func(func(p[0], p[2], s1), p[1], s2)
        bottom = func(func(p[6], p[8], s1), p[7], s2)
        gy = np.bitwise_xor(top, bottom)

        return func(gx, gy, s3)

    def correct(self, parr):
        mat = np.array(parr[0:9]).reshape(3, 3)
        Gx = np.sum(np.array([[1, 2, 1]]).T * mat * np.array([1, 0, -1]))
        Gy = np.sum(np.array([[1, 0, -1]]).T * mat * np.array([1, 2, 1]))
        return (1/8) * (np.abs(Gx) + np.abs(Gy))

class C_AND_OR(CombCirc):
    def __init__(self):
        super().__init__(2, 2, 0, [0,1], "And_Or")

    def run(self, bs_mat):
        return np.array([np.bitwise_and(bs_mat[0, :], bs_mat[1, :]), np.bitwise_or(bs_mat[0, :], bs_mat[1, :])])

    def correct(self, parr):
        return np.array([
            parr[0] * parr[1],
            parr[0] + parr[1] - parr[0] * parr[1]
        ])

def mux(x, y, s):
    return np.bitwise_or(
        np.bitwise_and(np.bitwise_not(s), x), 
        np.bitwise_and(s, y)
    )

def maj(x, y, s):
    return np.bitwise_or(
        np.bitwise_or(
            np.bitwise_and(s, x), 
            np.bitwise_and(s, y)
        ),
        np.bitwise_and(x, y)
    )

def mux_4_to_1(x1, x2, x3, x4, s1, s2):
    top = mux(x1, x2, s1)
    bot = mux(x3, x4, s1)
    return mux(top, bot, s2)

def robert_cross(x11, x22, x12, x21, s, is_maj=False):
    xor1, xor2 = np.bitwise_xor(x11, x22), np.bitwise_xor(x12, x21)
    if is_maj:
        return maj(xor1, xor2, s)
    else:
        return mux(xor1, xor2, s)

#def sobel(*x):
#    x = x[0:9]
#    s = x[9:12]
#
#    x11 = mux_4_to_1(s[0], s[1]) #TODO: finish
#    x22 = mux_4_to_1(s[0], s[1])
#    x12 = mux_4_to_1(s[0], s[1])
#    x21 = mux_4_to_1(s[0], s[1])
#    return robert_cross(x11, x22, x12, x21, s[2])