from abc import abstractmethod
import numpy as np
from functools import reduce
from sim.ReSC import ReSC, B_GAMMA

class Circ:
    def __init__(self, n, m, nc, cgroups, name):
        self.n = n #total number of inputs (including constant inputs)
        self.m = m #total number of outputs
        self.nc = nc #number of 0.5-valued constant inputs
        self.nv = self.n - self.nc
        self.cgroups = cgroups #code for correlated groups (including constants)
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

class C_WIRE(Circ):
    def __init__(self):
        super().__init__(1, 1, 0, [0,], "WIRE")

    def run(self, bs_mat):
        return bs_mat
    
    def correct(self, parr):
        return parr

class C_AND_N(Circ):
    def __init__(self, n):
        super().__init__(n, 1, 0, [x for x in range(n)], "AND Gate n={}".format(n))

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
    
class C_XOR(Circ):
    def __init__(self):
        super().__init__(2, 1, 0, [0, 1], "XOR Gate")

    def run(self, bs_mat):
        return np.bitwise_xor(bs_mat[0, :], bs_mat[1, :])
    
    def correct(self, parr):
        return parr[0] + parr[1] - 2*parr[0]*parr[1]
    
class C_MUX_ADD(Circ):
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
    
class C_MAC(Circ):
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
    
class C_RCED(Circ):
    def __init__(self):
        super().__init__(5, 1, 1, [0, 0, 0, 0], "RCED")

    def run(self, bs_mat):
        return robert_cross(*[bs_mat[x, :] for x in range(5)])

    def correct(self, parr):
        return 0.5 * (np.abs(parr[0] - parr[1]) + np.abs(parr[2] - parr[3]))
    
class C_Gamma(Circ):
    def __init__(self):
        super().__init__(13, 1, 0, [0, 1, 2, 3, 4, 5] + [6 for _ in range(7)], "ReSC Gamma")
        #Not sure how many 0.5-valued constants are required to generate the inputs to the ReSC circuit

    def parr_mod(self, parr):
        return np.array([parr[0],] * 6 + B_GAMMA)

    def run(self, bs_mat):
        return ReSC(bs_mat).flatten()
    
    def correct(self, parr):
        return parr[0] ** 0.45
    
class C_Sobel(Circ):
    def __init__(self):
        super().__init__(12, 1, 3, [0 for x in range(9)], "Sobel")
    
    def run(self, bs_mat):
        pass

    def correct(self, parr):
        mat = np.array(parr).reshape(3, 3)
        Gx = np.sum(np.array([[1, 2, 1]]).T * mat * np.array([1, 0, -1]))
        Gy = np.sum(np.array([[1, 0, -1]]).T * mat * np.array([1, 2, 1]))
        return (1/8) * (np.abs(Gx) + np.abs(Gy))
        #return min(np.abs(Gx) + np.abs(Gy), 1)

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