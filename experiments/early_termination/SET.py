import numpy as np
from sim.Util import clog2
from sim.SNG import *
from sim.circs.circs import *
from sim.SCC import scc
from experiments.early_termination.et_sim import gen_correct, ideal_SET, SET_hypergeometric
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

def hypergeo(N, p, Nmax):
    return (1/N) * p * (1-p) * (Nmax - N) / (Nmax - 1)

def test_basic_hypergeo(num):
    Nmax = 256
    Nrange = range(2, Nmax)
    errs = np.zeros((len(Nrange)))
    model_errs = np.zeros((len(Nrange)))
    for i in range(num):
        print(i)
        X = np.concatenate((np.ones(int(Nmax/2)), np.zeros(int(Nmax/2))))
        np.random.shuffle(X)
        for j, N in enumerate(Nrange):
            val = np.mean(X[:N])
            errs[j] += np.sqrt(MSE(0.5, val))
            model_errs[j] += np.sqrt(hypergeo(N, 0.5, Nmax))
    errs /= num
    model_errs /= num
    plt.plot(list(Nrange), errs)
    plt.plot(list(Nrange), model_errs, label="hypergeo")
    plt.title(r"Error $\epsilon$ vs. Bitstream length $N$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"Error: $\epsilon$")
    plt.legend()
    plt.show()

def err_vs_N_sweep(num):
    #This function is very similar to "fig_X() from early_termination_plots.py"
    w = 4
    n = 2
    circ = C_AND_N(2)
    Nmax = 2 ** (w * n)
    Nrange = range(2, Nmax)

    errs = np.zeros((len(Nrange)))
    model_errs = np.zeros((len(Nrange)))
    MA_model_errs = np.zeros((len(Nrange)))
    xs = [0.75, 0.75]
    xs = circ.parr_mod(xs)
    for i in range(num):
        print(i)
        bs_mat_full = true_rand_precise_sample(xs, w)
        #bs_mat_full = lfsr_sng_precise_sample(xs, w)
        for j, N in enumerate(Nrange):
            bs_mat = bs_mat_full[:, :N]

            out_val = np.mean(circ.run(bs_mat))
            correct = circ.correct(xs)
            #err_avg += np.sqrt(MSE(out_val, correct))
            errs[j] += np.sqrt(MSE(xs[0], np.mean(bs_mat, axis=1)[0]))
            var = hypergeo(N, correct, Nmax)

            #AND GATE
            var_x = hypergeo(N, xs[0], Nmax)
            var_y = hypergeo(N, xs[1], Nmax)
            model_errs[j] += np.sqrt(var_x)

            #MA_var = (1/N) * (xs[0] * (1-xs[0])) * (xs[1] * (1-xs[1])) * (Nmax - N) / (Nmax - 1)

            #Complete MA equation
            #MA_var = (1/(N-1)) * (xs[0] - var_x - xs[0] ** 2) * (xs[1] - var_y - xs[1] ** 2) + \
            #    var_x * xs[1] ** 2 + var_y * xs[0] ** 2 + var_x * var_y

            #MUX GATE
            #MA equation as defined in Tim's Hypergeometric Distribution Paper
            #MA_var = (1/N) * (xs[2] * (1 - xs[2])) * (xs[0] * (1 - xs[0]) + xs[1] * (1 - xs[1])) * (Nmax - N) / (Nmax - 1)

            #model_err_avg += np.sqrt(var)
            #MA_err_avg += np.sqrt(MA_var)

    errs /= num
    model_errs /= num
    MA_model_errs /= num

    print(xs)
    plt.plot(list(Nrange), errs)
    plt.plot(list(Nrange), model_errs, label="hypergeo")
    #plt.plot(list(Nrange), MA_model_errs, label="MA model")
    plt.title(r"Error $\epsilon$ vs. Bitstream length $N$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"Error: $\epsilon$")
    plt.legend()
    plt.show()

def err_vs_N_sweep_1(num):
    #This function is very similar to "fig_X() from early_termination_plots.py"
    w = 4
    n = 2
    circ = C_AND_N(2)
    Nmax = 2 ** (w * n)
    Nrange = range(2, 256)

    errs = []
    model_errs = []
    MA_model_errs = []
    xs = [0.5, 0.5]
    xs = circ.parr_mod(xs)
    for N in Nrange:
        print(N)
        err_avg = 0.0
        model_err_avg = 0.0
        MA_err_avg = 0.0
        #SCC_avg = 0.0
        for _ in range(num):
            bs_mat = true_rand_precise_sample(xs, w, Net=N)
            #bs_mat = lfsr_sng_precise_sample(xs, w, Net=N)

            #SCC_avg += np.abs(scc(bs_mat[0, :], bs_mat[1, :]))

            out_val = np.mean(circ.run(bs_mat))
            correct = circ.correct(xs)
            #err_avg += np.sqrt(MSE(out_val, correct))
            err_avg += np.sqrt(MSE(xs[0], np.mean(bs_mat, axis=1)[0]))
            var = hypergeo(N, correct, Nmax)

            #AND GATE
            var_x = hypergeo(N, xs[0], Nmax)
            var_y = hypergeo(N, xs[1], Nmax)
            model_err_avg += np.sqrt(var_x)

            #MA_var = (1/N) * (xs[0] * (1-xs[0])) * (xs[1] * (1-xs[1])) * (Nmax - N) / (Nmax - 1)

            #Complete MA equation
            #MA_var = (1/(N-1)) * (xs[0] - var_x - xs[0] ** 2) * (xs[1] - var_y - xs[1] ** 2) + \
            #    var_x * xs[1] ** 2 + var_y * xs[0] ** 2 + var_x * var_y

            #MUX GATE
            #MA equation as defined in Tim's Hypergeometric Distribution Paper
            #MA_var = (1/N) * (xs[2] * (1 - xs[2])) * (xs[0] * (1 - xs[0]) + xs[1] * (1 - xs[1])) * (Nmax - N) / (Nmax - 1)

            #model_err_avg += np.sqrt(var)
            #MA_err_avg += np.sqrt(MA_var)
        #print("SCC abs avg: ", SCC_avg / num)
        errs.append(err_avg / num)
        model_errs.append(model_err_avg / num)
        MA_model_errs.append(MA_err_avg / num)

    print(xs)
    plt.plot(list(Nrange), errs)
    plt.plot(list(Nrange), model_errs, label="hypergeo")
    #plt.plot(list(Nrange), MA_model_errs, label="MA model")
    plt.title(r"Error $\epsilon$ vs. Bitstream length $N$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"Error: $\epsilon$")
    plt.legend()
    plt.show()

def SET_px_sweep(num, threshs, Nmax = 256, runs = 21):
    plt.rcParams.update({'font.size': 13})
    pxs = np.linspace(0, 1, num)
    try:
        Ns = np.load("results/SET_px_sweep_{}_{}.npy".format(Nmax, runs))
    except:
        Ns = np.empty((len(threshs), num))
        for i, thresh in enumerate(threshs):
            for j, px in enumerate(pxs):
                print("{}/{}".format(j, num))
                N_trials = []
                for _ in range(runs):
                    bs_mat = true_rand_hyper_sng(np.array([px, ]), Nmax, clog2(Nmax), pack=False)
                    N_trials.append(optimal_ET(bs_mat, px, thresh))
                Ns[i, j] = np.mean(N_trials)
        np.save("results/SET_px_sweep_{}_{}.npy".format(Nmax, runs), Ns)
    for i, thresh in enumerate(threshs):
        plt.plot(pxs, Ns[i, :], label=r"$\epsilon_{var}$:" + " {}".format(thresh))
    plt.xlabel(r"$Z^*$")
    plt.ylabel(r"$N_{SET}$")
    plt.title(r"$N_{SET}$ as a function of $Z^*$, with $N_{max}=256$")
    plt.legend()
    plt.show()

    #This function only considers a single input directly generated from an LFSR, basically the simplest case possible
    #The hypergeometric model should predict the curves we get from this too

def SET_hypergeometric_px_sweep(num, threshs, Nmax = 256):
    plt.rcParams.update({'font.size': 13})
    pxs = np.linspace(0, 1, num)
    Ns = np.empty((len(threshs), num))

    a, b = 0.0362, 0.1817
    mnist = beta.pdf(pxs, a, b)[1:-1] / np.sum(beta.pdf(pxs, a, b)[1:-1])
    a, b = 3, 3
    center = beta.pdf(pxs, a, b)[1:-1] / np.sum(beta.pdf(pxs, a, b)[1:-1])

    #plt.plot(pxs[1:-1], mnist, label="MNIST")
    #plt.plot(pxs[1:-1], center, label="Normal")
    #plt.legend()
    #plt.xlabel(r"$Z^*$")
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
    plt.xlabel(r"$Z^*$")
    plt.ylabel(r"$N_{SET}$")
    plt.title(r"$N_{SET}$ as a function of $Z^*$, with $N_{max}=256$")
    plt.legend()
    plt.show()

def test_SET_hypergeo(num, thresh, runs = 100, Nmax = 256):
    pxs = np.linspace(0, 1, num)
    for px in pxs:
        N_et = np.round(SET_hypergeometric(px, thresh, Nmax=Nmax)).astype(np.int32)
        mse = 0
        for _ in range(runs):
            bs_mat = true_rand_hyper_sng(np.array([px, ]), Nmax, clog2(Nmax), pack=False)[:N_et]
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
                bs_mat = true_rand_hyper_sng(np.array([px, py]), Nmax ** 2, clog2(Nmax ** 2), pack=False)[:, :N_et]
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

def test_ideal_SET(ds, circ, e_min, e_max):
    w, Nmax, Nset = ideal_SET(ds, circ, e_min, e_max)
    correct_vals = gen_correct(ds, circ)
    cape_et_vals = []
    lfsr_et_vals = []
    for i, xs in enumerate(ds):
        if i % 100 == 0:
            print("{} out of {}".format(i, ds.shape[0]))

        xs = circ.parr_mod(xs) #Add constant inputs and/or duplicate certain inputs
        bs_mat = CAPE_sng(xs, w, circ.cgroups, Nmax=Nmax)[:, :Nset]
        bs_out_cape = circ.run(bs_mat)
        cape_et_vals.append(np.mean(bs_out_cape))

        bs_mat = lfsr_sng_efficient(xs, Nmax, w, cgroups=circ.cgroups, pack=False)[:, :Nset]
        bs_out_lfsr = circ.run(bs_mat)
        lfsr_et_vals.append(np.mean(bs_out_lfsr))

    print("Avg err CAPE: ", np.mean(np.abs(correct_vals - cape_et_vals)))
    print("Avg err LFSR: ", np.mean(np.abs(correct_vals - lfsr_et_vals)))