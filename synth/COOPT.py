import numpy as np
from sim.Util import int_array

#Legacy implementation of COMAX
def opt_K_max(K):
    _, tlen = K.shape
    K_sum = np.sum(K, axis=1).astype(np.int32)
    return np.stack([np.pad(np.ones(t, dtype=np.bool_), (0, tlen-t), 'constant') for t in K_sum])

def A_to_Mf(A, n, k):
    Mf = np.zeros((2**n, 2**k), dtype=np.bool_)
    for i in range(2**n):
        x = int_array(A[i, :].reshape(1, k))[0] #<-- why [0]
        Mf[i, x] = True
    return Mf

def Ks_to_A(Ks):
    nc, nv = np.log2(Ks[0].shape)
    n = int(nv + nc)
    k = len(Ks) #Ks is a list of K matrices
    A = np.zeros((2**n, k), dtype=np.bool_)
    for i, K in enumerate(Ks):
        A[:, i] = K.T.reshape(2**n)
    return A

def Ks_to_Mf(Ks):
    nc, nv = np.log2(Ks[0].shape)
    n = int(nv + nc)
    k = len(Ks) #Ks is a list of K matrices
    A = Ks_to_A(Ks)
    return A_to_Mf(A, n, k)