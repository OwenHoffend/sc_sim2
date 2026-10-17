from synth.experiments.example_circuits_for_proposal import *

def xor_and_copula():
    circ = XOR_with_AND()
    ptm = circ.get_PTM(lsb='left')
    T = copula_transform_matrix(ptm)
    print(T)