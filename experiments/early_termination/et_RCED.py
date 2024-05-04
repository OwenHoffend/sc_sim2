import numpy as np
from experiments.early_termination.et_hardware import *
from sim.circs import robert_cross

def RCED_et_kernel(px, max_precision, max_var, staticN=None):
    ctr_sz = max_precision
    lfsr_width = ctr_sz + 1
    if staticN is not None:
        Nmax = staticN
    else:
        Nmax = 2 ** lfsr_width

    #variance-based dynamic ET
    parr = np.concatenate((px, np.array([0.5,])))
    correct = 0.5 * (np.abs(px[0] - px[1]) + np.abs(px[2] - px[3]))
    cgroups = np.array([1, 1, 1, 1, 2])

    bs_mat = lfsr_sng_efficient(parr, Nmax, lfsr_width, cgroups=cgroups, pack=False)
    bs_out = np.zeros((Nmax,), dtype=np.bool_)
    for i in range(Nmax):
        bs_out[i] = robert_cross(*list(bs_mat[:, i]))

    pz_full = np.mean(bs_out)

    #N_et_var, cnts = var_et(bs_out, max_var)
    #pz_et_var = np.mean(bs_out[:N_et_var])

    #CAPE-based dynamic ET
    #cgroups = np.array([1, 1, 1, 1, 2])
    #bs_mat = CAPE_sng(parr, max_precision, cgroups, et=True, Nmax=Nmax)
    #_, N_et_CAPE = bs_mat.shape
    #bs_out = np.zeros((N_et_CAPE,), dtype=np.bool_)
    #for i in range(N_et_CAPE):
    #    bs_out[i] = robert_cross(*list(bs_mat[:, i]))
    #pz_et_CAPE = np.mean(bs_out)

    return correct, pz_full #, pz_et_var, N_et_var, pz_et_CAPE, N_et_CAPE

def RCED_et(max_precision, num_pxs, dist, staticN=None):
    #variance-based ET done using output samples only 
    nv = 4
    max_var = 0.01

    px_func = get_dist(dist, nv)

    ets_var = []
    ets_cape = []
    mses_var = []
    mses_cape = []
    for _ in range(num_pxs):
        print("-----")
        px = px_func()
        correct, pz_full, pz_et_var, N_et_var, pz_et_CAPE, N_et_CAPE = RCED_et_kernel(px, max_var, max_precision, staticN=staticN)
        
        et_mse_var = MSE(pz_et_var, correct)
        print("et mse var: ", et_mse_var)

        et_mse_CAPE = MSE(pz_et_CAPE, correct)
        print("et mse CAPE: ", et_mse_CAPE)

        ets_var.append(N_et_var)
        ets_cape.append(N_et_CAPE)
        mses_var.append(et_mse_var)
        mses_cape.append(et_mse_CAPE)

        #plt.plot(cnts)
        #plt.title("Counter value vs. N for true output variance: {} \n Terminated at: {} out of {}".format(np.round(pz * (1-pz), 3), N_et, 512))
        #plt.ylabel("Counter value")
        #plt.xlabel("N")
        #plt.show()

    print("AVG ET var: ", np.mean(ets_var))
    print("AVG ET CAPE: ", np.mean(ets_cape))
    print("AVG MSE var: ", np.mean(mses_var))
    print("AVG MSE CAPE: ", np.mean(mses_cape))

    return np.mean(ets_var), np.mean(ets_cape), np.mean(mses_var), np.mean(mses_cape)