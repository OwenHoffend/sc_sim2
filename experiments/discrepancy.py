import numpy as np
import matplotlib.pyplot as plt
import time
from sim.PTM import B_mat, get_func_mat

def get_possible_Ps(N):
    """Given a bitstream length N, get all of the possible probability values that can be made in a numpy array
    One thing worth noting is that this assumes all input probability values have an equal chance of occuring,
    One idea would be to adopt the Bayesian approach in combination with discepancy to improve circuit error analysis
    """
    ps = np.zeros(N, dtype=np.float32)
    for i in range(N):
        ps[i] = (i+1) / N
    return ps

class M_set():
    def __init__(self, s=None):
        if s is None:
            self.set = set()
        else:
            self.set = set(s)

    def add(self, s):
        self.set.add(s)

    def intersection(self, s):
        return M_set(self.set.intersection(s.set))

    def difference(self, s):
        return M_set(self.set.difference(s.set))
    
    def __mul__(self, s):
        return M_set(self.set.intersection(s.set))

def region_subsets(S, P, axis):
    N, _ = S.shape
    Sr = []
    Sr_r = []
    S_rsorted = S[S[:, axis].argsort()]
    start_idx = 0
    srp = M_set()
    all_points = M_set()
    for j in range(N):
        all_points.add(tuple(S[j, :]))

    for p in P:
        srp = M_set(srp.set)
        for j in range(start_idx, N):
            row = S_rsorted[j, :] 
            if row[axis] < p:
                srp.add(tuple(row))
            else:
                start_idx = j
                break
        Sr.append(srp)
        Sr_r.append(all_points.difference(srp))
    
    return Sr, Sr_r

def star_disc_2d(S, P): #has a worse time complexity than simulation, but better polynomial constant?
    """
    S is a numpy array of 2d points: [[x1, y1], [x2, y2], ..., [xN, yN]]
    P is a numpy array of all the probability values
    """
    N, _ = S.shape
    Sx, _ = region_subsets(S, P, 0)
    Sy, _ = region_subsets(S, P, 1)

    max_err = -1
    for i, px in enumerate(P):
        for j, py in enumerate(P):
            nb = len(Sx[i].intersection(Sy[j]))
            #print("i: {}, j: {}, nb: {}".format(i, j, nb))
            vol = px*py
            err = np.abs(nb/N - vol)
            if err > max_err:
                max_err = err
    return max_err

def star_disc_ptm(Mf, S, P):
    """
    S is a numpy array of d-dimensional points, of shape (N, d)
    P is a numpy array of all the possible probability values
    """
    N, d = S.shape
    n2, k2 = Mf.shape
    k = np.log2(k2).astype(int)
    A = B_mat(k).T @ Mf.T #Truth table matrix

    Srs = []
    Sr_rs = []
    for axis in range(d):
        Sr, Sr_r = region_subsets(S, P, axis)
        Srs.append(Sr)
        Sr_rs.append(Sr_r)

    all_points = M_set()
    for j in range(N):
        all_points.add(tuple(S[j, :]))
    
    max_err = -1
    def nestloop(depth, width, idx=[]):
        nonlocal max_err
        if(depth > 0):
            for i in range(width):
                nestloop(depth-1, width, idx=idx + [i, ])
        else:
            vin_approx_s = np.array([all_points,], dtype=object)
            for i in range(d):
                Sr = Srs[i][idx[i]]
                Sr_r = Sr_rs[i][idx[i]]
                vin_approx_s = np.kron(vin_approx_s, np.array([Sr_r, Sr]))
            vin_approx = np.zeros(n2)
            for i in range(n2):
                vin_approx[i] = len(vin_approx_s[i].set)
            vin_approx /= N ** d
            assert np.sum(vin_approx) == 1

            vin_ideal = np.array([1.0])
            Ps = P[idx]
            for i in range(d): #TODO: this is for independent inputs, generalize to any Cmat
                vin_ideal = np.kron(vin_ideal, np.array([1-Ps[i], Ps[i]]))
            err = np.abs(A @ (vin_approx - vin_ideal))
            if err > max_err:
                max_err = err
            
    nestloop(d, N)
    return max_err

def test_star_disc_2d():
    S = np.array([
        [0.125, 0.125],
        [0.25, 0.125],
        [0.4, 0.25],
        [0.8, 0.125],
        [0.25, 0.4],
        [0.4, 0.4],
        [0.5, 0.35],
        [0.9, 0.4],
        [0.125, 0.9],
        [0.125, 0.8],
        [0.4, 0.75],
        [0.5, 0.75],
        [0.8, 0.8]
    ])

    P = get_possible_Ps(3)
    print(star_disc_2d(S, P))

def test_star_disc_2d_with_lfsr():
    from sim.RNS import lfsr
    from sim.Util import bit_vec_to_int
    N = 64
    w = int(np.log2(N))

    #Get lfsr sequences and the input point cloud
    lfsr_seq_x = lfsr(w, N)
    lfsr_seq_y = lfsr(w, N)    
    S = np.zeros((N, 2))
    for i in range(N):
        S[i, 0] = bit_vec_to_int(lfsr_seq_x[:, i]) / N
        S[i, 1] = bit_vec_to_int(lfsr_seq_y[:, i]) / N

    #optionally plot the input point cloud
    #plt.scatter(S[:, 0], S[:, 1])
    #plt.show()

    P = get_possible_Ps(N)
    start = time.time()
    disc_result = star_disc_2d(S, P)
    end = time.time()
    print(disc_result)
    print("time: ", end - start)

    #simulate an AND-gate circuit
    start = time.time()
    max_err = -1
    for px in P:
        for py in P:
            pz = px * py
            pout = np.bitwise_and(S[:, 0] < px, S[:, 1] < py)
            err = np.abs(np.mean(pout) - pz)
            if err > max_err:
                max_err = err
    end = time.time()
    print(max_err)
    print("time: ", end - start)

def test_star_disc_ptm():
    from sim.RNS import lfsr
    from sim.Util import bit_vec_to_int
    N = 16
    w = int(np.log2(N))

    #Get lfsr sequences and the input point cloud
    lfsr_seq_x = lfsr(w, N)
    lfsr_seq_y = lfsr(w, N)    
    #lfsr_seq_z = lfsr(w, N)    
    S = np.zeros((N, 2))
    for i in range(N):
        S[i, 0] = bit_vec_to_int(lfsr_seq_x[:, i]) / N
        S[i, 1] = bit_vec_to_int(lfsr_seq_y[:, i]) / N
        #S[i, 2] = bit_vec_to_int(lfsr_seq_z[:, i]) / N

    #optionally plot the input point cloud
    #plt.scatter(S[:, 0], S[:, 1])
    #plt.show()

    test_ptm = get_func_mat(np.bitwise_and, 2, 1)
    P = get_possible_Ps(N)
    print(star_disc_ptm(test_ptm, S, P))