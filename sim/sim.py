import numpy as np
from sim.circs.circs import Circ
from sim.datasets import Dataset
from sim.SNG import SNG
from sim.PTM import get_PTM
from sim.PTV import get_PTV, get_C_from_v, get_Px_from_v
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
    
    def errs(self):
        return np.abs(self.correct - self.out)

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

def sim_circ(sng: SNG, circ: Circ, ds: Dataset, Nset=None, loop_print=True):
    if Nset is not None:
        N = np.round(Nset).astype(np.int32)
    else:
        N = circ.get_Nmax(sng.w)
    correct_vals = gen_correct(circ, ds) #ground truth output assuming floating point precision
    trunc_vals = gen_correct(circ, ds, trunc_w=sng.w) #ground truth output assuming w-bit fixed-point precision
    out = np.empty((ds.num, ))
    Ns = np.empty((ds.num, ))
    for i, xs in enumerate(ds):
        xs = circ.parr_mod(xs)
        if loop_print:
            print("{}/{}".format(i, ds.num))
        bs_mat = sng.run(xs, N)
        Nactual = bs_mat.shape[1]
        bs_out = circ.run(bs_mat)
        Z = np.mean(bs_out, axis=1 if circ.m > 1 else None)
        if hasattr(sng, "lzd_correction"):
            Z /= sng.lzd_correction
        out[i] = Z
        Ns[i] = Nactual
    return SimResult(correct_vals, trunc_vals, out, Ns)

def sim_circ_NSweep(sng: SNG, circ: Circ, ds: Dataset, Nrange: list, loop_print=True):
    #Nrange: list of N values to simulate the circuit with
    Nmax = max(Nrange)
    correct_vals = gen_correct(circ, ds) #ground truth output assuming floating point precision
    trunc_vals = gen_correct(circ, ds, trunc_w=sng.w) #ground truth output assuming w-bit fixed-point precision
    out = np.empty((ds.num, len(Nrange)))
    for i, xs in enumerate(ds):
        xs = circ.parr_mod(xs)
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

def sim_circ_PTM(circ: Circ, ds: Dataset, Cin, validate=False, lsb='right'):
    Mf = get_PTM(circ, lsb=lsb)

    Cs = []
    correct_vals = []
    Ps = []
    v_consts = get_PTV(np.identity(circ.nc), np.array([0.5 for _ in range(circ.nc)]), lsb=lsb)
    for i, xs in enumerate(ds):
        P_correct = circ.correct(xs)
        correct_vals.append(P_correct)

        vin = get_PTV(Cin, xs, lsb=lsb)

        #add nc uncorrelated constants to the PTV
        if lsb == 'left':
            vin = np.kron(v_consts, vin)
        else:
            vin = np.kron(vin, v_consts)

        vout = Mf.T @ vin
        P, Cout = get_C_from_v(vout, return_P=True, lsb=lsb)


        #Compare to correct output probabilities (sanity check)
        #TODO: Currently circ.correct does not take into account the input correlation matrix
        if validate:
            #print(f"Actual: {P}, Correct: {P_correct}")
            assert np.all(np.isclose(P, P_correct))

        Cs.append(Cout)
        Ps.append(P)

    #print(Cs)
    return SimResult(correct_vals, None, Ps, None)

cache: dict = {}
def gen_correct(circ: Circ, ds: Dataset, trunc_w=None, use_cache=False):
    """
    For a given dataset, produce the set of correct output values given the provided stochastic circuit
    circ: instance of a stochastic circuit to evaluate; must implement the "circ.correct" method
    ds: instance of a 
    """
    assert circ.m < 2 #FIXME: I think this code breaks when the circuit has more than 1 output

    global cache
    if use_cache:
        if trunc_w is not None:
            if trunc_w in cache:
                return cache[trunc_w]
        elif -1 in cache:
            return cache[-1]

    correct_vals = []
    for xs in ds:
        if trunc_w is not None:
            xs = list(map(lambda px: np.floor(px * 2 ** trunc_w) / (2 ** trunc_w), xs))
        correct_vals.append(circ.correct(xs))
    result = np.array(correct_vals).flatten()
    
    if use_cache:
        if trunc_w is not None:
            cache[trunc_w] = result
        else:
            cache[-1] = result
    return result