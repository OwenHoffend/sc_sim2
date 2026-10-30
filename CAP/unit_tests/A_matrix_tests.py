import numpy as np
from CAP.PTV import *

def get_random_LD_matrix(n):
    A_nonnorm = np.triu(np.random.rand(n, n))
    A = A_nonnorm / np.sum(A_nonnorm, axis=0)
    s = (np.random.rand(n) > 0.5) * 2 - 1
    S = np.diag(s)
    C = S @ (A @ A.T * (np.ones((n,n))-np.eye(n)) + np.eye(n)) @ S
    print(C)
    return A, s, C

def A_matrix_tests():
    #C1 = np.array([
    #    [1, -0.4, 0.3], 
    #    [-0.4, 1, -0.3], 
    #    [0.3, -0.3, 1]
    #])
    #C2 = np.array([
    #    [1, 0, -1],
    #    [0, 1, 0],
    #    [-1, 0, 1]
    #])
    #from scipy.linalg import block_diag
    #C = block_diag(C1, C2)
    #print(C)
#
    #s, A = get_acoeffs_and_signs_from_C(C)
    #print(s)
#
    np.set_printoptions(suppress=True, precision=3)
    #print(A.T)

    #test_nonint_scc_gen_3vars()

    n = 10
    for i in range(1000):
        A, s, C = get_random_LD_matrix(n)
        print(C)

        s2, A2 = get_acoeffs_and_signs_from_C(C)
        S2 = np.diag(s2)
        C2 = S2 @ (A2 @ A2.T * (np.ones((n,n))-np.eye(n)) + np.eye(n)) @ S2

        print(C2)
        if not np.allclose(C, C2, atol=0.1):
            print("BAD")
