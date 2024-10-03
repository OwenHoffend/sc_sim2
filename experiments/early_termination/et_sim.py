from sim.RNS import *
from experiments.early_termination.et_hardware import *

def ET_sim(ds, circ, e_min, e_max, SET_override=None, j=0):
    w, Nmax, Nset = ideal_SET(ds, circ, e_min, e_max)
    _, Nmin, _ = N_from_trunc_err(ds, circ, e_max)
    #_, Nmax, _ = N_from_trunc_err(ds, circ, e_max)
    w = clog2(Nmax)
    correct_vals = gen_correct(ds, circ)

    if SET_override is not None:
        Nmax, Nset, Nmin = SET_override

    SET_vals = []
    vret_vals = []
    pret_vals = []

    vret_Ns = []
    pret_Ns = []
    for i, xs in enumerate(ds):
        if i % 100 == 0:
            print("{} out of {}".format(i, ds.shape[0]))

        xs = circ.parr_mod(xs) #Add constant inputs and/or duplicate certain inputs

        #CAPE+VRET performance
        bs_mat = CAPE_sng(xs, clog2(Nmin), circ.cgroups, et=True, Nmax=Nmin, use_consensus_for_corr=False)
        bs_out_cape = circ.run(bs_mat)

        pret_vals.append(np.mean(bs_out_cape))
        pret_Ns.append(bs_out_cape.size)
        #Nvret = var_et_new(bs_out_cape, e_max, power_of_2=True)
        #pret_vals.append(np.mean(bs_out_cape[:Nvret]))
        #pret_Ns.append(Nvret)

        #Baseline SC performance
        bs_mat = true_rand_sng_efficient(xs, Nmax, w, cgroups=circ.cgroups, pack=False)
        #bs_mat = CAPE_sng(xs, w, circ.cgroups, et=False, Nmax=Nmax)
        bs_out_sc = circ.run(bs_mat)
        SET_vals.append(np.mean(bs_out_sc[:Nset]))

        #VRET performance
        Nvret = var_et_new(bs_out_sc, (e_max - e_min) ** 2, power_of_2=True)
        #Nvret = var_et(bs_out_sc, (e_max - e_min) ** 2, power_of_2=True)
        vret_vals.append(np.mean(bs_out_sc[:Nvret]))
        vret_Ns.append(Nvret)

    #print stats

    err_set = np.mean(np.abs(correct_vals - SET_vals))
    err_vret = np.mean(np.abs(correct_vals - vret_vals))
    err_pret = np.mean(np.abs(correct_vals - pret_vals))
    N_vret = np.mean(vret_Ns)
    N_pret = np.mean(pret_Ns)
    print("j: {}, Nset: {}".format(j, Nset))
    print("{} Avg err SET: {}".format(j, err_set))
    print("{} Avg err VRET: {}".format(j, err_vret))
    print("{} Avg err PRET+VRET: {}".format(j, err_pret))

    print("{} Avg N: SET: {}".format(j, Nset))
    print("{} Avg N: VRET: {}".format(j, N_vret))
    print("{} Avg N: PRET+VRET: {}".format(j, N_pret))

    return (j, err_set, err_vret, err_pret, N_vret, N_pret)

def ET_sim_2(correct_vals, ds, circ, Nmin, Nset, Nmax, e_min, e_max):
    SET_vals = []
    vret_vals = []
    pret_vals = []

    vret_Ns = []
    pret_Ns = []
    for i, xs in enumerate(ds):
        if i % 100 == 0:
            print("{} out of {}".format(i, ds.shape[0]))

        xs = circ.parr_mod(xs) #Add constant inputs and/or duplicate certain inputs

        #CAPE+VRET performance
        bs_mat = CAPE_sng(xs, clog2(Nmin), circ.cgroups, et=True, Nmax=Nmin, use_consensus_for_corr=False)
        bs_out_cape = circ.run(bs_mat)

        pret_vals.append(np.mean(bs_out_cape))
        pret_Ns.append(bs_out_cape.size)
        #Nvret = var_et_new(bs_out_cape, e_max, power_of_2=True)
        #pret_vals.append(np.mean(bs_out_cape[:Nvret]))
        #pret_Ns.append(Nvret)

        #Baseline SC performance
        bs_mat = true_rand_sng_efficient(xs, Nmax, clog2(Nmax), cgroups=circ.cgroups, pack=False)
        #bs_mat = CAPE_sng(xs, w, circ.cgroups, et=False, Nmax=Nmax)
        bs_out_sc = circ.run(bs_mat)
        SET_vals.append(np.mean(bs_out_sc[:Nset]))

        #VRET performance
        Nvret = var_et_new(bs_out_sc, (e_max - e_min) ** 2, power_of_2=True)
        #Nvret = var_et(bs_out_sc, (e_max - e_min) ** 2, power_of_2=True)
        vret_vals.append(np.mean(bs_out_sc[:Nvret]))
        vret_Ns.append(Nvret)

    #print stats

    err_set = np.mean(np.abs(correct_vals - SET_vals))
    err_vret = np.mean(np.abs(correct_vals - vret_vals))
    err_pret = np.mean(np.abs(correct_vals - pret_vals))
    N_vret = np.mean(vret_Ns)
    N_pret = np.mean(pret_Ns)

    return err_set, err_vret, err_pret, N_vret, N_pret

def SET_hypergeometric(pz, err_thresh, Nmax = 256):
    mse_thresh = err_thresh ** 2
    return (Nmax * pz * (1-pz)) / (mse_thresh * Nmax - mse_thresh + pz * (1-pz))

def N_from_trunc_err(ds, circ, e_min):
    correct_vals = gen_correct(ds, circ)
    w = 1
    while True:
        trunc_vals = gen_correct(ds, circ, trunc_w=w)
        e_trunc = np.mean(np.abs(correct_vals - trunc_vals))
        Nmax = circ.get_Nmax(w)
        print("e_trunc: ", e_trunc)
        if e_trunc <= e_min:
            break
        w += 1
    print(Nmax)
    return w, Nmax, trunc_vals


def ideal_SET(ds, circ, e_min, e_max):
    #Step 1: Compute Nmax based on the minimum error bound, e_min
    #strategy: try every Nmax until one that meets the threshold is found
    w, Nmax, trunc_vals = N_from_trunc_err(ds, circ, e_min)

    #Step 2: Compute Net based on the maximum error bound, e_max
    #cutiererererererererererererer
    e_var = e_max - e_min
    Nets = [SET_hypergeometric(pz_trunc, e_var, Nmax=Nmax) for pz_trunc in trunc_vals]
    Nset = np.ceil(np.mean(Nets)).astype(np.int32)

    #find the closest power of 2
    Nset = 2 ** clog2(Nset)
    print(Nset)

    return w, Nmax, Nset

def gen_correct(dataset, circ, trunc_w=None):
    correct_vals = []
    for xs in dataset:
        if trunc_w is not None:
            xs = list(map(lambda px: np.floor(px * 2 ** trunc_w) / (2 ** trunc_w), xs))
        correct_vals.append(circ.correct(xs))
    correct_vals = np.array(correct_vals).flatten()
    return correct_vals