import numpy as np

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


def et_var_test(num_trials, N, hypergeometric=False):
    nv2 = 128
    nc2 = 128
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
    MSE = 0
    order = np.array([x for x in range(nc2)])
    np.random.shuffle(order)
    for _ in range(num_trials):
        samples = np.random.choice(nv2, N, p=vin)
        out = 0.0
        for i in range(N):
            ws = w[samples[i]]
            if hypergeometric:
                if (order[i] % nc2) / nc2 <= ws:
                    out += 1.0
            else:
                out += np.random.choice(np.array([0.0, 1.0]), 1, p=np.array([1-ws, ws]))
        out /= N
        MSE += (out - correct) ** 2
    MSE /= num_trials
    print("MSE: ", MSE)
    print("var: ", var)

def dynamic_et_test():
    nv2 = 4
    nc2 = 4
    var = 0.01
    Nmax = np.ceil(1.0 / 4*var).astype(int)
    vin = np.random.uniform(size=(nv2,))
    vin = vin / np.sum(vin)
    w = np.random.uniform(0, nc2-1, size=(nv2,)) / nc2
    correct = vin.T @ w

    num_pmf_vals = 20
    pmf_vals = np.linspace(0, 1, num_pmf_vals)
    dims = tuple([num_pmf_vals,] * nv2)
    init_uniform = 1.0 / (num_pmf_vals ** nv2)
    v_pmf = np.full(dims, init_uniform)

    for t in range(Nmax):
        pass