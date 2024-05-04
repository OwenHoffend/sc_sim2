from experiments.early_termination.et_hardware import *
from experiments.early_termination.precision_analysis import used_prec
from sim.deterministic import rotation_2d, clock_division_2d

def x_squared_et(w):
    num_trials = 100
    max_var = 0.001
    full = []
    var = []
    var_prec = []
    cape = []
    LD = []

    Ns_var = []
    Ns_var_prec = []
    Ns_cape = []
    Ns_LD = []

    Nmax = 2 ** w
    xvals = np.array([x/256 for x in range(1, 256)])
    for px in xvals:
        cgroups = np.array([0, 1])
        parr = np.array([px, px])

        #Normal SC
        N_lfsr = 256
        bs_mat = lfsr_sng_efficient(parr, N_lfsr, w, cgroups=cgroups, pack=False)
        bs_out = np.zeros((N_lfsr,), dtype=np.bool_)
        for i in range(N_lfsr):
            bs_out[i] = np.bitwise_and(bs_mat[0, i], bs_mat[1, i])
        pz_full = np.mean(bs_out)
        full.append(pz_full)

        #print("LFSR SCC: ", scc_mat(bs_mat)[1, 0])

        #Variance-based ET
        N_et, cnts = var_et(bs_out, max_var)
        pz_var = np.mean(bs_out[:N_et])
        var.append(pz_var)
        Ns_var.append(N_et)

        #Precision-informed variance-based ET
        Nprec = 2 ** (2 * used_prec(px, 4))
        N_et, cnts = var_et(bs_out, max_var, Nprec=Nprec)
        pz_var = np.mean(bs_out[:N_et])
        var_prec.append(pz_var)
        Ns_var_prec.append(N_et)

        #CAPE SNG
        bs_mat = CAPE_sng(parr, w, cgroups, et=True, Nmax=Nmax)
        _, N_et_CAPE = bs_mat.shape
        bs_out = np.zeros((N_et_CAPE,), dtype=np.bool_)
        for i in range(N_et_CAPE):
            bs_out[i] = np.bitwise_and(bs_mat[0, i], bs_mat[1, i])
        pz_et_CAPE = np.mean(bs_out)
        cape.append(pz_et_CAPE)
        Ns_cape.append(N_et_CAPE)

        #LD precision-based
        N_et_LD = 2 ** (2 * used_prec(px, 4))
        ld_rns = clock_division_2d(van_der_corput, 4, 2 ** used_prec(px, 4))
        pbin = parr_bin(parr, 4, lsb="left")
        pbin_ints = int_array(pbin)
        bs_mat = np.zeros((2, N_et_LD), dtype=np.bool_)
        for i in range(2):
            for j in range(N_et_LD):
                bs_mat[i, j] = pbin_ints[i] > ld_rns[i][j]
        bs_out = np.zeros((N_et_LD,), dtype=np.bool_)
        for i in range(N_et_LD):
            bs_out[i] = np.bitwise_and(bs_mat[0, i], bs_mat[1, i])
        pz_et_LD = np.mean(bs_out)
        LD.append(pz_et_LD)
        Ns_LD.append(N_et_LD)

        #1: Don't know why the CAPE/LD ET lengths are a bit longer than what's predicted by the theory
            #This might actually be because the input distribution is not uniform, rather both inputs receive the same value
        #2: Need to investigate why it doesn't seem like low-precision values actual have any ET benefit with LFSRs
            #LFSRs don't necessarily guarantee perfect sampling at bitstream lengths less than their full periods
            #Early terminating LFSR sequences based on precision does not work


    correct = xvals ** 2
    print("SC MSE: ", MSE(correct, full))
    print("Var MSE: ", MSE(correct, var))
    print("Var prec MSE: ", MSE(correct, var_prec))
    print("CAPE MSE: ", MSE(correct, cape))
    print("LD MSE: ", MSE(correct, LD))

    print("Var ET: ", np.mean(Ns_var))
    print("Var prec ET: ", np.mean(Ns_var_prec))
    print("CAPE ET: ", np.mean(Ns_cape))
    print("LD ET: ", np.mean(Ns_LD))

    plt.plot(xvals, xvals ** 2, label="correct")
    plt.plot(xvals, full, label="full")
    plt.plot(xvals, var, label="var")
    plt.plot(xvals, cape, label="cape")
    plt.plot(xvals, LD, label="LD")
    plt.xlabel("Px")
    plt.ylabel("Py")
    plt.title("Squarer ET performance")
    plt.legend()
    plt.show()

def mul3_et_kernel(px, max_precision, max_var, staticN=None, only_sc=False):
    lfsr_width = max_precision
    if staticN is not None:
        Nmax = staticN
    else:
        Nmax = 2 ** (lfsr_width * 3)

    #variance-based dynamic ET
    correct = px[0] * px[1] * px[2]
    cgroups = np.array([1, 2, 3])

    bs_mat = lfsr_sng_efficient(px, Nmax, lfsr_width, cgroups=cgroups, pack=False)
    bs_out = np.zeros((Nmax,), dtype=np.bool_)
    for i in range(Nmax):
        bs_out[i] = np.bitwise_and(np.bitwise_and(bs_mat[0, i], bs_mat[1, i]), bs_mat[2, i])

    pz_full = np.mean(bs_out)

    if only_sc:
        return correct, pz_full

    N_et_var, cnts = var_et(bs_out, max_var)
    pz_et_var = np.mean(bs_out[:N_et_var])

    #CAPE-based dynamic ET
    bs_mat = CAPE_sng(px, max_precision, cgroups, et=True, Nmax=Nmax)
    _, N_et_CAPE = bs_mat.shape
    bs_out = np.zeros((N_et_CAPE,), dtype=np.bool_)
    for i in range(N_et_CAPE):
        bs_out[i] = np.bitwise_and(np.bitwise_and(bs_mat[0, i], bs_mat[1, i]), bs_mat[2, i])
    pz_et_CAPE = np.mean(bs_out)

    #Both techniques
    N_et_both, cnts = var_et(bs_out, max_var)
    pz_et_both = np.mean(bs_out[:N_et_both])

    return correct, pz_full, pz_et_var, pz_et_both, N_et_var, pz_et_CAPE, N_et_CAPE, N_et_both