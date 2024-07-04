from sim.RNS import *
from experiments.early_termination.et_hardware import *

def ET_sim(circ, dataset, Nmax, w, max_var=0.001):
    SC_vals = []
    var_et_vals = []
    var_et_Ns = []
    cape_et_vals = []
    cape_et_Ns = []
    LD_et_vals = []
    LD_et_Ns = []
    for i, xs in enumerate(dataset):
        if i % 100 == 0:
            print("{} out of {}".format(i, dataset.shape[0]))

        xs = circ.parr_mod(xs) #Add constant inputs and/or duplicate certain inputs

        #Baseline LFSR-based SC performance
        bs_mat = lfsr_sng_efficient(xs, Nmax, w, cgroups=circ.cgroups, pack=False)
        bs_out_sc = circ.run(bs_mat)
        SC_vals.append(np.mean(bs_out_sc))

        #Variance-based ET performance
        N_et_var = var_et(bs_out_sc, max_var, exact=False, power_of_2=True)
        bs_out_var = bs_out_sc[:N_et_var]
        var_et_vals.append(np.mean(bs_out_var))
        var_et_Ns.append(N_et_var)

        #CAPE-without ET performance (Low discrepancy generator)
        #bs_mat = CAPE_sng(xs, w, circ.cgroups, et=False, Nmax=Nmax)
        #bs_out_LD = circ.run(bs_mat)
        #LD_et_vals.append(np.mean(bs_out_LD))
        #LD_et_Ns.append(bs_out_LD.size)

        #CAPE-based ET performance
        bs_mat = CAPE_sng(xs, w, circ.cgroups, et=True, Nmax=Nmax)
        bs_out_cape = circ.run(bs_mat)
        cape_et_vals.append(np.mean(bs_out_cape))
        cape_et_Ns.append(bs_out_cape.size)

    return SC_vals, var_et_vals, var_et_Ns, cape_et_vals, cape_et_Ns, LD_et_vals, LD_et_Ns

def gen_correct(dataset, circ, trunc_w=None):
    correct_vals = []
    for xs in dataset:
        if trunc_w is not None:
            xs = list(map(lambda px: np.floor(px * 2 ** trunc_w) / (2 ** trunc_w), xs))
        correct_vals.append(circ.correct(xs))
    correct_vals = np.array(correct_vals).flatten()
    return correct_vals