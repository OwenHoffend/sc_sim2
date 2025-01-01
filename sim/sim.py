import numpy as np
from sim.circs.circs import Circ
from sim.datasets import Dataset
from sim.SNG import SNG
from sim.Util import MSE

class SimResult:
    def __init__(self, xs, correct, trunc, out, N):
        self.xs = xs
        self.correct = correct
        self.trunc = trunc
        self.out = out
        self.N = N

class SimRun:
    def __init__(self, correct, trunc, results: list[SimResult]):
        self.correct = correct
        self.trunc = trunc
        self.results = results

    def RMSE_vs_N(self):
        Nvals = {}
        for result in self.results:
            mse = MSE(result.out, result.correct)
            if result.N not in Nvals:
                Nvals[result.N] = [mse, ]
            else:
                Nvals[result.N].append(mse)

        Ns = []
        rmses = []
        for k, v in Nvals.items():
            Ns.append(k)
            rmses.append(np.sqrt(np.mean(v)))
        return Ns, rmses
    
    def avg_N(self):
        #FIXME: doesn't really make sense when using Nrange 

        Ntot = 0
        for result in self.results:
            Ntot += result.N
        return Ntot / len(self.results)
    
    def RMSE(self):
        #FIXME: doesn't really make sense when using Nrange

        err_total = 0
        for result in self.results:
            err_total += MSE(result.out, result.correct)
        return np.sqrt(err_total / len(self.results))

def sim_circ(sng: SNG, circ: Circ, ds: Dataset, Nrange: list | None = None, loop_print=True):
    #Nrange: list of N values to simulate the circuit with

    #derive Nrange from circ using the normal method if not explicitly provided
    if Nrange is None:
        Nmax = circ.get_Nmax(sng.w)
    else:
        Nmax = max(Nrange)

    correct_vals = gen_correct(circ, ds) #ground truth output assuming floating point precision
    trunc_vals = gen_correct(circ, ds, trunc_w=sng.w) #ground truth output assuming w-bit fixed-point precision
    sim_results: list[SimResult] = []
    for i, xs in enumerate(ds):
        if loop_print:
            print(i)
        bs_mat_full = sng.run(xs, Nmax)
        Nret = bs_mat_full.shape[1]
        if Nrange is None:
            _Nrange = [Nret, ]
        else:
            _Nrange = Nrange

        for N in _Nrange:
            bs_mat = bs_mat_full[:, :N]
            bs_out = circ.run(bs_mat)
            Z = np.mean(bs_out, axis=1 if circ.m > 1 else None)
            if hasattr(sng, "lzd_correction"):
                Z /= sng.lzd_correction
            sim_results.append(SimResult(xs, correct_vals[i], trunc_vals[i], Z, N))

    return SimRun(correct_vals, trunc_vals, sim_results)

def gen_correct(circ: Circ, ds: Dataset, trunc_w=None):
    """
    For a given dataset, produce the set of correct output values given the provided stochastic circuit
    circ: instance of a stochastic circuit to evaluate; must implement the "circ.correct" method
    ds: instance of a 
    """
    assert circ.m < 2 #FIXME: I think this code breaks when the circuit has more than 1 output

    correct_vals = []
    for xs in ds:
        if trunc_w is not None:
            xs = list(map(lambda px: np.floor(px * 2 ** trunc_w) / (2 ** trunc_w), xs))
        correct_vals.append(circ.correct(xs))
    return np.array(correct_vals).flatten()