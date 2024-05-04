from sim.ReSC import *
from experiments.early_termination.et_hardware import var_et
from experiments.early_termination.precision_analysis import used_prec

def gamma_correction(num_samples=25, w=8, Nmax=256, max_var=0.001):

    #When I limit to Nmax, this is effectively a static early termination

    x_vals = np.linspace(0, 1, num_samples)

    #bbin = parr_bin(np.array(b), w, lsb="left")
    y_vals = np.zeros((num_samples,))
    y_vals_et = np.zeros((num_samples), )
    y_vals_CAPE_et = np.zeros((num_samples), )
    y_vals_prec_et_LFSR = np.zeros((num_samples), )
    savings = []
    savings_CAPE = []
    savings_prec = []
    for idx, x in enumerate(x_vals):
        #print(idx)

        #Variance-based early termination:
        #Get the input bitstreams
        parr = np.array([x,] * 6 + B_GAMMA)
        cgroups = np.array([0, 1, 2, 3, 4, 5] + [6 for _ in range(7)])
        bs_mat = lfsr_sng(parr, Nmax, w, cgroups=cgroups, pack=False)
        bs_out = ReSC(bs_mat).flatten()

        N_et_var, _ = var_et(bs_out, max_var)
        savings.append(N_et_var)
        y_vals[idx] = np.mean(bs_out)
        y_vals_et[idx] = np.mean(bs_out[:N_et_var])

        #CAPE-based early termination:
        #bs_mat = CAPE_sng(parr, w, cgroups, Nmax=Nmax, et=True)
        #bs_out_cape = ReSC(bs_mat).flatten()
        #y_vals_CAPE_et[idx] = np.mean(bs_out_cape)
        #savings_CAPE.append(bs_out_cape.size)

        #Precision-based early termination using the LFSR bits as a source
        actual_req_prec = used_prec(x, 7) + max([used_prec(a, 7) for a in B_GAMMA])
        N_prec_et = min(2 ** actual_req_prec, Nmax)
        y_vals_prec_et_LFSR[idx] = np.mean(bs_out[:N_prec_et])
        print(N_prec_et)

        y_vals_prec_et_LFSR[idx] = np.mean(bs_out[:N_prec_et])
        
    plt.title("Gamma correction Early Termination Test \n Avg. var et length: {}/1024 \n Avg. CAPE et length: {}"
              .format(np.mean(np.array(savings)), np.mean(np.array(savings_CAPE))))
    #plt.plot(x_vals, x_vals ** 0.45, label="correct")
    #plt.plot(x_vals, bernstein(np.array(B_GAMMA), x_vals), label="bernstein approx")
    plt.plot(x_vals, y_vals, label="SC, N={}".format(Nmax))
    plt.plot(x_vals, y_vals_prec_et_LFSR, label="SC, prec-based ET")
    #plt.plot(x_vals, y_vals_et, label="SC, var-based ET")
    #plt.plot(x_vals, y_vals_CAPE_et, label="SC, CAPE-based ET")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    print(MSE(y_vals, x_vals ** 0.45))
    print(MSE(y_vals_et, x_vals ** 0.45))
    print(MSE(y_vals_prec_et_LFSR, x_vals ** 0.45))

    #plt.plot(x_vals, savings)
    #plt.plot(x_vals, savings_CAPE)
    #plt.title("Actual bits sampled for each X value")
    #plt.show()

def gamma_correction_imgs():
    pass