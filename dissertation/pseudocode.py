import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal
from statsmodels.distributions.copula.api import GaussianCopula

def scc(px, py, pxy):
    cov = pxy - px * py
    if cov > 0:
        return cov / (min(px, py) - px * py)
    else:
        return cov / (px * py - max(px + py - 1, 0))

def binary_vector(value, m):
    """Convert a positive integer value into an m-bit bit vector"""
    arr = np.array(
        list(np.binary_repr(value).zfill(m))
    ).astype(bool)

    return arr

def CAP(Cin, Px, Mf, mode="auto"):
    n = np.log2(Mf.shape[0])
    m = np.log2(Mf.shape[1])
    copula_func = get_copula_func(Cin, Px, mode=mode)
    pin = get_MV_from_copula(copula_func, Px)
    Qn = get_Q(n)
    Qm = get_Q(m)
    pout = Qm @ Mf @ np.linalg.inv(Qn) @ pin
    return get_C_and_Px_from_MV(pout, m)

def get_Q(n):
    Q0 = np.array([[1, 1], [0, 1]])
    Qn = Q0
    for _ in range(1, n):
        Qn = np.kron(Qn, Q0)
    return Qn

def get_copula_func(Cin, Px, mode="auto"):
    pass

def get_FH_copula_func(XR):
    """XR specifies the RNS structure. Its structure is:
        XR[0][i] is the set of variables (indices) that use RNS Ri
        XR[1][i] is the set of variables (indices) that use the inverted RNS Ri'
        This implements Eq. 21 in the paper
    """
    def FH_C(Px):
        n = len(Px)
        c = 1
        for i in range(n):
            PXR = [Px[j] for j in XR[0][i]]
            PXR_inv = [Px[j] for j in XR[1][i]]
            c *= max(min(PXR) + min(PXR_inv) - 1, 0)
        return c
    return FH_C

def get_Gaussian_copula_func(Cin, num_restarts=100):
    """
    Parameters
    ----------
    Cin : np.ndarray
        Target SCC matrix.
    Px : np.ndarray
        Marginal probabilities P(X_i = 1).
    num_restarts : int
        Number of random initializations.

    Returns
    -------
    GC: A Gaussian copula function that can be called
    """


    """First fit a Gaussian copula to the desired correlation and marginals"""
    n = Cin.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def xvec_to_Pearson(x):
        B = x.reshape(n, n)

        norms = np.linalg.norm(B, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)

        B /= norms
        R = B @ B.T
        return R

    def GC(Px):

        def objective(x):
            R = xvec_to_Pearson(x)

            err = 0.0
            for i, j in pairs:

                pi, pj = Px[i], Px[j]
                rho = np.clip(R[i, j], -0.999999, 0.999999)

                ti = norm.ppf(pi)
                tj = norm.ppf(pj)

                cov = np.array([
                    [1.0, rho],
                    [rho, 1.0],
                ])

                p_ij = multivariate_normal.cdf(
                    [ti, tj],
                    mean=[0.0, 0.0],
                    cov=cov,
                )

                c_hat = scc(pi, pj, p_ij)
                diff = c_hat - Cin[i, j]
                err += diff * diff

            return err

        best_result = None
        best_value = np.inf

        rng = np.random.default_rng(0)

        for _ in range(num_restarts):
            x0 = rng.normal(size=(n, n)).ravel()

            result = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                options={
                    "maxiter": 1000,
                    "ftol": 1e-12,
                },
            )

            if result.fun < best_value:
                best_result = result
                best_value = result.fun

        R_best = xvec_to_Pearson(best_result.x, n, n)
        R_best = 0.5 * (R_best + R_best.T) #FIXME: not sure what this line does
        np.fill_diagonal(R_best, 1.0)
        return GaussianCopula(R_best, allow_singular=True).cdf

    return GC

def get_MV_from_copula(copula_func, Px):
    n = len(Px)
    pin = np.empty((2 ** n, ))
    for i in range(2 ** n):
        Px_star = np.empty((n, ))
        b = binary_vector(i, n)
        for j in range(n):
            Px_star[j] = Px[n-1-j] if b[j] else 1
        pin[i] = copula_func(*Px_star)
    return pin

def get_C_and_Px_from_MV(p, n):
    C = np.zeros((n, n))
    Px = np.zeros((n,))
    for i in range(n):
        p_idx_i = 2 ** (n - 1 - i) #switch to 2 ** n for P(X_2, X_1, X_0 = 0,0,1) indexing
        pi = p[p_idx_i]
        Px[i] = pi
        for j in range(n):
            p_idx_j = 2 ** (n - 1 - j)
            p_idx_ij = p_idx_i + p_idx_j
            pj = p[p_idx_j]
            pij = p[p_idx_ij]
            C[i, j] = scc(pi, pj, pij)
    return C, Px