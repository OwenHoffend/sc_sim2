import numpy as np
from sim.circs.circs import Circ
from sim.datasets import Dataset
from sim.SNG import SNG
from sim.Util import MSE

class SimResult:
    def __init__(self, correct, trunc, out, Nrange):
        self.correct = correct
        self.trunc = trunc
        self.out = out
        self.Nrange = Nrange

        #number of dataset inputs, number of N values, number of circuit outputs
        self.num, self.Ncnt, self.m = out.shape

    def RMSE_vs_N(self):
        errs = np.zeros((len(self.Nrange)))
        for i, correct in enumerate(self.correct):
            for j in range(self.Ncnt):
                errs[j] += MSE(self.out[i, j], correct)
        return np.sqrt(errs / self.num)

def sim_circ(sng: SNG, circ: Circ, ds: Dataset, Nrange: list | range | None = None, loop_print=True):
    #Nrange: list of N values to simulate the circuit with

    #derive Nrange from circ using the normal method if not explicitly provided
    if Nrange is None:
        Nmax = circ.get_Nmax(sng.w)
        Nrange = range(2, Nmax + 1)
    else:
        Nmax = max(Nrange)

    correct_vals = gen_correct(circ, ds) #ground truth output assuming floating point precision
    trunc_vals = gen_correct(circ, ds, trunc_w=sng.w) #ground truth output assuming w-bit fixed-point precision
    out_vals = np.zeros((ds.num, len(Nrange), circ.m))
    for i, xs in enumerate(ds):
        if loop_print:
            print(i)
        bs_mat_full = sng.run(xs, Nmax)
        for j, N in enumerate(Nrange):
            bs_mat = bs_mat_full[:, :N]
            bs_out = circ.run(bs_mat)
            out_vals[i, j, :] = np.mean(bs_out, axis=1 if circ.m > 1 else None)

    return SimResult(correct_vals, trunc_vals, out_vals, Nrange)

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