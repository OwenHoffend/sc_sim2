import numpy as np
from sim.Util import clog2
from sim.SNG import *
from experiments.early_termination.et_sim import gen_correct
from scipy.stats import beta
import matplotlib.pyplot as plt

def optimal_ET(bs, correct, thresh):
    #Given a bitstream, the correct value, and an error threshold, finds the shortest bitstream length that meets the theshold
    running_sum = 0
    max_N = 0
    for i in range(bs.size):
        running_sum += bs[i]
        Pn = running_sum / (i + 1)
        #print(np.abs(Pn - correct))
        if np.abs(Pn - correct) > thresh:
            max_N = i + 1
    #print(max_N)
    return max_N

def SET_px_sweep(num, threshs, Nmax = 32, runs = 21):
    pxs = np.linspace(0, 1, num)
    #try:
        #Ns = np.load("results/SET_px_sweep_{}_{}.npy".format(Nmax, runs))
    #except:
    Ns = np.empty((len(threshs), num))
    for i, thresh in enumerate(threshs):
        for j, px in enumerate(pxs):
            print("{}/{}".format(j, num))
            N_trials = []
            for _ in range(runs):
                bs_mat = true_rand_sng(np.array([px, ]), Nmax, clog2(Nmax), pack=False)
                N_trials.append(optimal_ET(bs_mat, px, thresh))
            Ns[i, j] = np.mean(N_trials)
    for i, thresh in enumerate(threshs):
        plt.plot(pxs, Ns[i, :], label="Thresh: {}".format(thresh))
    np.save("results/SET_px_sweep_{}_{}.npy".format(Nmax, runs), Ns)
    plt.xlabel(r"$P^*_Z$")
    plt.ylabel(r"$N_{SET}$")
    plt.title(r"$N_{SET}$ as a function of $P^*_Z$, with $N_{max}=256$")
    plt.legend()
    plt.show()

    #This function only considers a single input directly generated from an LFSR, basically the simplest case possible
    #The hypergeometric model should predict the curves we get from this too

def SET_hypergeometric(pz, err_thresh, Nmax = 256):
    mse_thresh = err_thresh ** 2
    return (Nmax * pz * (1-pz)) / (mse_thresh * Nmax - mse_thresh + pz * (1-pz))

def SET_hypergeometric_px_sweep(num, threshs, Nmax = 256):
    pxs = np.linspace(0, 1, num)
    Ns = np.empty((len(threshs), num))

    a, b = 0.0362, 0.1817
    mnist = beta.pdf(pxs, a, b)[1:-1] / np.sum(beta.pdf(pxs, a, b)[1:-1])
    a, b = 3, 3
    center = beta.pdf(pxs, a, b)[1:-1] / np.sum(beta.pdf(pxs, a, b)[1:-1])

    #plt.plot(pxs[1:-1], mnist, label="MNIST beta")
    #plt.plot(pxs[1:-1], center, label="Center beta")
    #plt.legend()
    #plt.xlabel(r"$P_Z$")
    #plt.ylabel("Density")
    #plt.title("Value distribution example")
    #plt.show()

    for i, thresh in enumerate(threshs):
        for j, px in enumerate(pxs):
            #print("{}/{}".format(j, num))
            N_trials = []
            N_trials.append(SET_hypergeometric(px, thresh, Nmax=Nmax))
            Ns[i, j] = np.mean(N_trials)
        plt.plot(pxs, Ns[i, :], label=r"$\epsilon_{var}$:" + " {}".format(thresh))
        print(mnist @ Ns[i, 1:-1])
        print(center @ Ns[i, 1:-1])

    np.save("results/SET_hypergeo_sweep_{}.npy".format(Nmax), Ns)
    plt.xlabel(r"$P^*_Z$")
    plt.ylabel(r"$N_{SET}$")
    plt.title(r"$N_{SET}$ as a function of $P^*_Z$, with $N_{max}=256$")
    plt.legend()
    plt.show()

def test_SET_hypergeo(num, thresh, runs = 100, Nmax = 256):
    pxs = np.linspace(0, 1, num)
    for px in pxs:
        N_et = np.round(SET_hypergeometric(px, thresh, Nmax=Nmax)).astype(np.int32)
        mse = 0
        for _ in range(runs):
            bs_mat = true_rand_sng(np.array([px, ]), Nmax, clog2(Nmax), pack=False)[:N_et]
            err = (np.mean(bs_mat) - px) ** 2 
            mse += err
        mse /= runs
        print("Px: {}, N_et: {}, RMSE: {}".format(px, N_et, np.sqrt(mse)))

#PICKUP POINT:
#The next step is to test this setup on the output from a multi-input circuit
#to ensure my assumptions about how ET works with multi-input circuits are valid

#For example Nmax = 2 ** (w * s + nc)
#Then we use the output distribution of Pz combined with the baseline hypergeometric curves

def test_SET_hypergeo_2input(num, thresh, runs = 100, Nmax = 256):
    pxs = np.linspace(0, 1, num)
    for px in pxs:
        for py in pxs:
            correct = px * py #replace with the correct function for the chosen 2-input circuit
            N_et = np.round(SET_hypergeometric(correct, thresh, Nmax=Nmax ** 2)).astype(np.int32)
            if N_et == 0:
                N_et = 1
            mse = 0
            for _ in range(runs):
                bs_mat = true_rand_sng(np.array([px, py]), Nmax ** 2, clog2(Nmax ** 2), pack=False)[:, :N_et]
                output = np.bitwise_and(bs_mat[0, :], bs_mat[1, :])
                err = (np.mean(output) - correct) ** 2 
                mse += err
            mse /= runs
            print("Px: {}, Py: {}, N_et: {}, RMSE: {}".format(px, py, N_et, np.sqrt(mse)))

def error_breakdown(runs, Nmax = 64):
    w = clog2(np.sqrt(Nmax))
    quant_err_1_array = []
    corr_err_array = []
    var_err_array = []
    total_err_array = []
    for N_et in range(1, Nmax):
        total_err = quant_err_1 = quant_err_2 = corr_err = var_err = 0.0
        for _ in range(runs):
            px = np.random.uniform()
            py = np.random.uniform()
            px_trunc = fp_array(p_bin(px, w, lsb="right"))
            correct = px * py #replace with the correct function for the chosen 2-input circuit
            py_trunc = fp_array(p_bin(py, w, lsb="right"))
            trunc_correct = px_trunc * py_trunc

            bs_mat = CAPE_sng(np.array([px, py]), w, [0, 1]) #CAPE used here because it has precise sampling
            bs_mat_et = bs_mat[:, :N_et]
            full_output = np.mean(np.bitwise_and(bs_mat[0, :], bs_mat[1, :]))
            et_output = np.mean(np.bitwise_and(bs_mat_et[0, :], bs_mat_et[1, :]))
            
            total_err += np.abs(et_output - correct)
            quant_err_1 += np.abs(full_output - correct)
            quant_err_2 += np.abs(trunc_correct - correct)
            corr_err += np.abs(et_output - np.mean(bs_mat_et[0, :]) * np.mean(bs_mat_et[1, :]))

            hypergeo = (1/N_et) * (trunc_correct) * (1-trunc_correct) * ((Nmax - N_et)/(Nmax - 1))
            var_err += np.sqrt(hypergeo)

        quant_err_1_array.append(quant_err_1 / runs)
        corr_err_array.append(corr_err / runs)
        var_err_array.append(var_err / runs)
        total_err_array.append(total_err / runs) # if you need the total error

    # Create a stacked area chart
    plt.stackplot(range(1, Nmax), quant_err_1_array, corr_err_array, var_err_array,
                labels=['Quantization Error', 'Correlation Error', 'Variance Error'])

    # Labels and titles
    plt.title('Error Analysis Over N_et')
    plt.xlabel('N_et')
    plt.ylabel('Error Value')
    plt.legend(loc='upper right')

    # Displaying the plot
    plt.show()

def quant_error(num, ws):
    pxs = np.linspace(0, 1, num)

    for w in ws:
        errs = []
        for px in pxs:
            #px_trunc = fp_array(p_bin(px, w, lsb="right"))
            px_trunc = np.floor(px * 2 ** w) / (2 ** w)
            errs.append(np.abs(px - px_trunc))
        plt.plot(pxs, errs, label="Width: {}".format(w))
    plt.show()

def ideal_SET(ds, circ, e_min, e_max):
    #Step 1: Compute Nmax based on the minimum error bound, e_min
    #strategy: try every Nmax until one that meets the threshold is found
    correct_vals = gen_correct(ds, circ)
    w = 1
    while True:
        trunc_vals = gen_correct(ds, circ, trunc_w=w)
        e_trunc = np.mean(np.abs(correct_vals - trunc_vals))
        print(e_trunc)
        if e_trunc <= e_min:
            break
        w += 1
    Nmax = 2 ** ((circ.n - circ.nc) * w + circ.nc)
    print(Nmax)

    #Step 2: Compute Net based on the maximum error bound, e_max
    #cutiererererererererererererer
    e_var = e_max - e_min
    Nets = [SET_hypergeometric(pz_trunc, e_var, Nmax=Nmax) for pz_trunc in trunc_vals]
    Nset = np.ceil(np.mean(Nets)).astype(np.int32)
    print(Nset)

    return w, Nmax, Nset

def test_ideal_SET(ds, circ, e_min, e_max):
    pass