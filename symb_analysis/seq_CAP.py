import sympy as sp
import numpy as np
from sim.PTV import get_Q

def get_steady_state(T, vars=None):
    eigs = (T.T).eigenvects()

    #Find the eigenvector with eigenvalue closest to 1
    eigvals = np.array([x[0] for x in eigs])
    if vars is not None:
        eigval_idx = None
        for i in range(len(eigs)):
            if eigvals[i].subs(sum(vars), 1) == 1:
                eigval_idx = i
                break
        if eigval_idx is None:
            raise ValueError("No eigenvector with eigenvalue closest to 1 found")
        eigvec_indx = eigval_idx
    else:
        eigvec_indx = np.argmin(np.abs(eigvals - 1))
    eigvec = eigs[eigvec_indx][2][0]

    #Normalize the eigenvector
    eigvec = eigvec / np.sum(eigvec)
    return sp.simplify(eigvec)

def transition_matrix_to_FSM(num_states, T):
    pass

def FSM_to_transition_matrix(num_states, transitions):
    #Given a specification of an FSM, return its transition matrix
    #(0, -1, x(1-y))
    T = sp.zeros(3, 3)
    for transition in transitions:
        pass