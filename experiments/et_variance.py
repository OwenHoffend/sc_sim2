import numpy as np
import matplotlib.pyplot as plt
from sim.PTM import get_vin_mc0, get_vin_mc1, get_vin_mcn1
from sim.Util import avg_loop, array_loop

def stupid_var_test(num_trials, N):
    p = np.random.uniform()
    var = p * (1-p) / N
    MSE = 0
    for i in range(num_trials):
        out = 0.0
        for j in range(N):
            out += np.random.choice(np.array([0.0, 1.0]), 1, p=np.array([1-p, p]))
        out /= N
        MSE += (out - p) ** 2
    MSE /= num_trials
    print("MSE: ", MSE)
    print("var: ", var)

def get_MSE_binomial(w, vin, N, nv2, num_trials):
    #Helper function to compute the actual output variance
    MSE = 0
    correct = vin.T @ w
    #order = np.array([x for x in range(nc2)])
    #np.random.shuffle(order)
    for _ in range(num_trials):
        samples = np.random.choice(nv2, N, p=vin)
        out = 0.0
        for i in range(N):
            ws = w[samples[i]]
            #if hypergeometric:
            #    if (order[i] % nc2) / nc2 <= ws:
            #        out += 1.0
            #else:
            out += np.random.choice(np.array([0.0, 1.0]), 1, p=np.array([1-ws, ws]))
        out /= N
        MSE += (out - correct) ** 2
    MSE /= num_trials
    return MSE

def et_var_test(num_trials, N, hypergeometric=False):
    nv2 = 4
    nc2 = 4
    vin = np.random.uniform(size=(nv2,))
    vin = vin / np.sum(vin)
    w = np.random.uniform(0, nc2-1, size=(nv2,)) / nc2
    correct = vin.T @ w

    if hypergeometric:
        assert N <= nc2 #Hypergeometric variance eq. can't really handle cases where N > nc2
    
    #Compute correct MSE based on the equation
    if hypergeometric:
        lhs = vin.T @ (w * (1 - w) * ((nc2 - N) / (nc2 - 1)) + w ** 2)
        var = lhs - correct ** 2
    else:
        var = correct *  (1 - correct)
    var /= N

    #Compute the actual output variance
    MSE = get_MSE_binomial(w, vin, N, nv2, num_trials)
    print("MSE: ", MSE)
    print("var: ", var)

def bayes_biased_coin(bias):
    #Using Bayes theorem, construct a PDF representing our knowledge of the bias parameter of a biased coin

    Nmax = 100
    num_pmf_vals = 100
    v_pmf = np.full((num_pmf_vals, ), 1.0 / num_pmf_vals)
    pmf_vals = np.linspace(0, 1, num_pmf_vals)

    for t in range(Nmax):
        sample = np.random.choice(2, 1, p=np.array([1-bias, bias]))[0]
        likelihood = (pmf_vals ** sample) * ((1-pmf_vals) ** (1-sample))
        evidence = likelihood.T @ v_pmf
        v_pmf = likelihood.T * v_pmf / evidence
    plt.plot(pmf_vals, v_pmf)
    plt.show()

def binomial_dynamic_et_test(w, vin, var, num_trials, plot=False):
    #Dynamic ET test that seeks to estimate the variable-input PMF by simply summing
    #each input pattern and dividing by t

    nv2 = w.size
    Nmax = np.ceil(1 / (4*var)).astype(int)
    et_ideal = np.ceil(vin.T @ w * (1-(vin.T @ w))/var).astype(int)
    v_pmf_int = np.ones(nv2) / nv2

    N_ests = []
    et = Nmax
    for t in range(Nmax):
        sample = np.zeros(nv2)
        sample[np.random.choice(nv2, 1, p=vin)[0]] = 1
        v_pmf_int += sample
        v_pmf = v_pmf_int / (t+2)
        est = v_pmf.T @ w
        N_est = np.ceil((est * (1-est)) / var).astype(int)
        N_ests.append(N_est)
        if t >= N_est and t < et:
            et = t
    print("Early terminate at: {} instead of {}".format(et, Nmax))

    MSE = get_MSE_binomial(w, vin, et, nv2, num_trials)
    if MSE > 2*var:
        print("WARNING: HIGH MSE")
        print("MSE: ", MSE)
        print("var: ", var)

    if plot:
        plt.title("MUX ET Plot for \n vin={}, var={}".format(list(vin), var))
        plt.plot(np.array([x for x in range(Nmax)]), label="t (clock cycle)")
        plt.plot(np.array([et_ideal] * Nmax), label="Dynamic ET Estimate")
        plt.plot(N_ests, label="Static ET Estimate")
        plt.scatter(np.array([et]),np.array([et]), s=25, c='red', label="Dynamic ET point: N_d={}".format(et))
        plt.scatter(np.array([et_ideal]),np.array([et_ideal]), s=25, c='purple', label="Ideal ET point: N_i={}".format(et_ideal))
        plt.legend()
        plt.xlabel("Current Timestep, t")
        plt.ylabel("N, number of bits")
        plt.show()

    return et / Nmax

def scc_dynamic_et_test(w, var, num_pxs, num_trials, corr, dist):
    nv2 = w.size
    n = np.ceil(np.log2(nv2)).astype(int)

    if dist == 'uniform':
        px_func = lambda: np.random.uniform(size=n)
    elif dist == 'MNIST_beta':
        px_func = lambda: np.random.beta(0.0362, 0.1817, size=n) #From Tim Baker's "Bayesian Analysis" paper
    elif dist == 'center_beta':
        px_func = lambda: np.random.beta(3, 3, size=n)
    else:
        raise ValueError

    def inner():
        px = px_func()
        if corr == 0:
            vin = get_vin_mc0(px)
        elif corr == 1:
            vin = get_vin_mc1(px)
        elif corr == -1:
            while np.sum(px) > 1:
                px = px_func()
            vin = get_vin_mcn1(px)
        else:
            raise ValueError
        return binomial_dynamic_et_test(w, vin, var, num_trials)

    vals = array_loop(inner, num_pxs)
    plt.hist(vals, bins=20, range=(0.0, 1.0))
    plt.title("Dynamic ET test: SCC={}, var={}, dist={}, \n avg={}, std={}".format(corr, var, dist, 
                                                                    np.round(np.mean(vals), 2), np.round(np.std(vals), 2)))
    plt.ylim((0, 300))
    plt.show()

#OLD CODE
#def dynamic_et_test():
#    nv2 = 4
#    nc2 = 4
#    Nmax = 100
#    vin = np.array([0.5, 0, 0.5, 0])
#    vin = vin / np.sum(vin)
#    #w = np.random.uniform(0, nc2-1, size=(nv2,)) / nc2
#    #correct = vin.T @ w
#
#    num_pmf_vals = 20
#    pmf_vals_1d = np.linspace(0, 1, num_pmf_vals)
#    
#    init_uniform = 1.0 / (num_pmf_vals ** nv2)
#    dims = tuple([num_pmf_vals,] * nv2)
#    v_pmf = np.full(dims, init_uniform)
#
#    for t in range(Nmax):
#        sample = np.random.choice(nv2, 1, p=vin)[0]
#        likelihood = pmf_vals_nv2d[sample]
#        evidence = np.sum(likelihood * v_pmf)
#        v_pmf = likelihood * v_pmf / evidence
#
#    #marginal distribution for the first pmf entry:
#    plt.plot(pmf_vals_1d, np.sum(v_pmf, axis=(1, 2, 3)))
#    plt.show()


