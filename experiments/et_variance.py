import numpy as np
import matplotlib.pyplot as plt
from sim.PTM import get_vin_mc0, get_vin_mc1, get_vin_mcn1
from sim.Util import avg_loop, array_loop, clog2

def bitstream_var_test_binomial(num_trials, N):
    N1 = np.random.randint(0, N)
    p = N1 / N
    var = (p * (1-p)) / N
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

def bitstream_var_test_hypergeometric(num_trials, N, nc2):
    assert N < nc2
    W1 = np.random.randint(0, nc2)
    w = W1 / nc2
    var = (w * (1-w) / N) * ((nc2 - N) / (nc2 - 1))
    MSE = 0
    for i in range(num_trials):
        bs = np.concatenate((np.ones(W1,), np.zeros((nc2-W1,))))
        np.random.shuffle(bs)
        out = np.mean(bs[:N])
        MSE += (out - w) ** 2
    MSE /= num_trials
    print("MSE: ", MSE)
    print("var: ", var)

def get_MSE_binomial(w, vin, N, nv2, num_trials):
    #Helper function to compute the actual output variance
    MSE = 0
    correct = vin.T @ w
    for _ in range(num_trials):
        samples = np.random.choice(nv2, N, p=vin)
        out = 0.0
        for i in range(N):
            ws = w[samples[i]]
            out += np.random.choice(np.array([0.0, 1.0]), 1, p=np.array([1-ws, ws]))
        out /= N
        MSE += (out - correct) ** 2
    MSE /= num_trials
    return MSE

def get_MSE_hypergeometric(w, vin, N, nv2, nc2, m, num_trials):
    MSE = 0
    correct = vin.T @ w
    var = (correct * (1-correct) / N) * ((m*nc2 - N) / (m*nc2 - 1))

    F = np.zeros((nv2, nc2))
    for i, weight in enumerate(w):
        W1 = (weight * nc2).astype(int)
        F[i, :] = np.concatenate((np.ones(W1,), np.zeros((nc2-W1,))))


    vin_ints = vin * m
    v_inds = []
    for idx, vi in enumerate(vin_ints):
        v_inds += [idx, ] * vi.astype(int)
    v_inds = np.array(v_inds)
    c_inds = np.array(range(nc2))
    v, c = np.meshgrid(v_inds, c_inds)
    v_inds = v.flatten()
    c_inds = c.flatten()

    assert v_inds.size == m*nc2
    assert c_inds.size == m*nc2

    trials = np.empty(num_trials)
    mnc2i = (m*nc2).astype(int)
    order = np.array(range(mnc2i))
    for t in range(num_trials):
        #print(t)
        out = 0.0
        np.random.shuffle(order)
        for i in range(N):
            o = order[i]
            vidx = v_inds[o % mnc2i]
            cidx = c_inds[o % mnc2i]
            out += F[vidx, cidx]

        out /= N
        trials[t] = out
        MSE += (out - correct) ** 2
    MSE /= num_trials
    print("MSE: ", MSE)
    print("actual var: ", np.var(trials))
    print("model var: ", var)
    return MSE

def et_var_test(num_trials, N, hypergeometric=False):

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

def get_dist(dist, n):
    if dist == 'uniform':
        px_func = lambda: np.random.uniform(size=n)
    elif dist == 'MNIST':
        px_func = lambda: sample_from_mnist(np.sqrt(n).astype(int))
    elif dist == 'MNIST_beta':
        px_func = lambda: np.random.beta(0.0362, 0.1817, size=n)
    elif dist == 'center_beta':
        px_func = lambda: np.random.beta(3, 3, size=n)
    else:
        raise ValueError
    return px_func

def static_et_eval(w, var, num_pxs, dist):
    #Compute the best N to early terminate at based on dataset statistics
    nv2 = w.size
    n = clog2(nv2)
    px_func = get_dist(dist, n)

    for i in range(num_pxs):
        pass

def binomial_static_et_test(w, vin, var, num_trials):
    pass

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
        plt.title("RCED ET Plot for \n px11=px12=0.1, px21=px22=0.9, scc=0, var={}".format(var))
        plt.plot(np.array([x for x in range(Nmax)]), label="t (clock cycle)")
        plt.plot(np.array([et_ideal] * Nmax), label="Static ET Estimate")
        plt.plot(N_ests, label="Dynamic ET Estimate")
        plt.scatter(np.array([et]),np.array([et]), s=25, c='red', label="Dynamic ET point: N_d={}".format(et))
        plt.scatter(np.array([et_ideal]),np.array([et_ideal]), s=25, c='purple', label="Ideal ET point: N_i={}".format(et_ideal))
        plt.legend()
        plt.xlabel("Current Timestep, t")
        plt.ylabel("N, number of bits")
        plt.show()

    return et / Nmax

def sample_from_mnist(winsz):
    mnist = np.load("C:/Users/owenh/OneDrive - Umich/research/code/sc_sim2/experiments/train_images.npy")
    n, _ = mnist.shape
    img = np.random.randint(0, n)
    xloc = np.random.randint(winsz, 28-winsz) #not near the edge
    yloc = np.random.randint(winsz, 28-winsz) #not near the edge
    return (mnist[img, :].reshape(28, 28))[yloc:yloc+winsz, xloc:xloc+winsz].reshape(winsz ** 2) / 255

def scc_dynamic_et_test(w, var, num_pxs, num_trials, corr, dist):
    nv2 = w.size
    n = clog2(nv2)

    px_func = get_dist(dist, n)

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
                                                                    np.round(np.mean(vals), 3), np.round(np.std(vals), 3)))
    plt.ylim((0, num_pxs))
    plt.show()