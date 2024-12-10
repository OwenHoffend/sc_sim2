import numpy as np
from sim.circs.circs import Circ
from sim.datasets import Dataset

def sim_circ(circ: Circ, ds: Dataset):
    pass

def gen_correct(circ: Circ, ds: Dataset, trunc_w=None):
    """
    For a given dataset, produce the set of correct output values given the provided stochastic circuit
    circ: instance of a stochastic circuit to evaluate; must implement the "circ.correct" method
    ds: instance of a 
    """
    correct_vals = []
    for xs in ds:
        if trunc_w is not None:
            xs = list(map(lambda px: np.floor(px * 2 ** trunc_w) / (2 ** trunc_w), xs))
        correct_vals.append(circ.correct(xs))
    return np.array(correct_vals).flatten()