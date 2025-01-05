import numpy as np
from sim.circs.circs import Circ
from sim.datasets import Dataset
from sim.SNG import SNG
from sim.Util import MSE

class SimResult:
    """Sim run assuming only one trial per correct value - N sweep will be done separately"""

    def __init__(self, correct, trunc, out, Ns):
        self.correct = correct
        self.trunc = trunc
        self.out = out
        self.Ns = Ns

    def RMSE_vs_N(self):
        """For runtime early termination where N is expected to vary between runs"""
        Nvals = {}
        for i, out in enumerate(self.out):
            mse = MSE(out, self.correct[i])
            currentN = self.Ns[i]
            if currentN not in Nvals:
                Nvals[currentN] = [mse, ]
            else:
                Nvals[currentN].append(mse)

        Ns = []
        rmses = []
        for k, v in Nvals.items():
            Ns.append(k)
            rmses.append(np.sqrt(np.mean(v)))
        return Ns, rmses
    
    def avg_N(self):
        return np.mean(self.Ns)
    
    def RMSE(self):
        err_total = 0
        for i, out in enumerate(self.out):
            err_total += MSE(out, self.correct[i])
        return np.sqrt(err_total / len(self.correct))

class NSweepSimResult:
    def __init__(self, correct, trunc, out, Ns):
        self.correct = correct
        self.trunc = trunc
        self.out = out
        self.Ns = Ns

    def RMSE_vs_N(self):
        num_Ns = len(self.Ns)
        rmses = np.zeros((num_Ns, ))
        for i, correct in enumerate(self.correct):
            for j in range(num_Ns):
                rmses[j] += MSE(self.out[i, j], correct)
        return self.Ns, np.sqrt(rmses / len(self.correct))

def sim_circ(sng: SNG, circ: Circ, ds: Dataset, loop_print=True):
    Nmax = circ.get_Nmax(sng.w)
    correct_vals = gen_correct(circ, ds) #ground truth output assuming floating point precision
    trunc_vals = gen_correct(circ, ds, trunc_w=sng.w) #ground truth output assuming w-bit fixed-point precision
    out = np.empty((ds.num, ))
    Ns = np.empty((ds.num, ))
    for i, xs in enumerate(ds):
        if loop_print:
            print("{}/{}".format(i, ds.num))
        bs_mat = sng.run(xs, Nmax)
        Nret = bs_mat.shape[1]
        bs_out = circ.run(bs_mat)
        Z = np.mean(bs_out, axis=1 if circ.m > 1 else None)
        if hasattr(sng, "lzd_correction"):
            Z /= sng.lzd_correction
        out[i] = Z
        Ns[i] = Nret
    return SimResult(correct_vals, trunc_vals, out, Ns)

def sim_circ_NSweep(sng: SNG, circ: Circ, ds: Dataset, Nrange: list, loop_print=True):
    #Nrange: list of N values to simulate the circuit with
    Nmax = max(Nrange)
    correct_vals = gen_correct(circ, ds) #ground truth output assuming floating point precision
    trunc_vals = gen_correct(circ, ds, trunc_w=sng.w) #ground truth output assuming w-bit fixed-point precision
    out = np.empty((ds.num, len(Nrange)))
    for i, xs in enumerate(ds):
        if loop_print:
            print("{}/{}".format(i, ds.num))
        bs_mat_full = sng.run(xs, Nmax)
        Nret = bs_mat_full.shape[1]
        if Nret != Nmax:
            raise NotImplementedError("NSweep simulation not implemented for RET")

        for j, N in enumerate(Nrange):
            bs_mat = bs_mat_full[:, :N]
            bs_out = circ.run(bs_mat)
            Z = np.mean(bs_out, axis=1 if circ.m > 1 else None)
            out[i, j] = Z
    return NSweepSimResult(correct_vals, trunc_vals, out, Nrange)

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