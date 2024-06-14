import numpy as np
from sim.circs import *
from sim.datasets import *
from img.img_quality import *
from experiments.early_termination.early_termination_plots import *

def conf_mat_bsds500_sobel(bsds_id, load=False):
    if load:
        correct_vals = np.load("results/bsds500_sobel_{}_correct.npy".format(bsds_id))
    else:
        circ = C_Sobel()
        ds = dataset_single_image("../datasets/bsds500/images/test/{}.jpg".format(bsds_id), 3)
        correct_vals = []
        for xs in ds:
            correct_vals.append(circ.correct(xs))
        correct_vals = np.array(correct_vals).flatten().reshape(321, 481)
        np.save("results/bsds500_sobel_{}_correct.npy".format(bsds_id), correct_vals)

    np.save("../edge_eval_python/examples/eval-result/sobel/{}.npy".format(bsds_id), correct_vals)

def conf_mat_bsds500_rced(bsds_id, load=False, external=True):
    if load:
        correct_vals = np.load("results/bsds500_{}_correct.npy".format(bsds_id))
        SC_vals = np.load("results/bsds500_{}_SC.npy".format(bsds_id))
        cape_et_vals = np.load("results/bsds500_{}_cape.npy".format(bsds_id))
        var_et_vals = np.load("results/bsds500_{}_var.npy".format(bsds_id))
        cape_et_Ns = np.load("results/bsds500_{}_cape_Ns.npy".format(bsds_id))
        var_et_Ns = np.load("results/bsds500_{}_var_Ns.npy".format(bsds_id))
        static_et_vals = np.load("results/bsds500_{}_static.npy".format(bsds_id))
    else:
        circ = C_RCED()
        ds = dataset_single_image("../datasets/bsds500/images/test/{}.jpg".format(bsds_id), 2)
        correct_vals = []
        for xs in ds:
            correct_vals.append(circ.correct(xs))
        correct_vals = np.array(correct_vals).flatten()
        np.save("results/bsds500_{}_correct.npy".format(bsds_id), correct_vals.reshape(320, 480) * 256)

        max_var = 0.001
        SC_vals, var_et_vals, var_et_Ns, cape_et_vals, cape_et_Ns, LD_et_vals, LD_et_Ns \
        = ET_sim(circ, ds, 64, 6, max_var=max_var)

        np.save("results/bsds500_{}_SC.npy".format(bsds_id), np.array(SC_vals).reshape(320, 480) * 256)
        np.save("results/bsds500_{}_cape.npy".format(bsds_id), np.array(cape_et_vals).reshape(320, 480) * 256)
        np.save("results/bsds500_{}_var.npy".format(bsds_id), np.array(var_et_vals).reshape(320, 480) * 256)
        np.save("results/bsds500_{}_cape_Ns.npy".format(bsds_id), np.array(cape_et_Ns).reshape(320, 480))
        np.save("results/bsds500_{}_var_Ns.npy".format(bsds_id), np.array(var_et_Ns).reshape(320, 480))

        static_et_vals = []
        for i, xs in enumerate(ds):
            if i % 100 == 0:
                print("{} out of {}".format(i, ds.shape[0]))
            xs = circ.parr_mod(xs) #Add constant inputs and/or duplicate certain inputs
            bs_mat = lfsr_sng_efficient(xs, 16, 6, cgroups=circ.cgroups, pack=False)
            bs_out_sc = circ.run(bs_mat)
            static_et_vals.append(np.mean(bs_out_sc))
        np.save("results/bsds500_{}_static.npy".format(bsds_id), np.array(static_et_vals).reshape(320, 480) * 256)

    if external:
        def pad_(v):
            return np.pad(v / 256, ((1, 0), (1, 0))).reshape(321, 481)
        np.save("../edge_eval_python/examples/eval-result/rced/{}.npy".format(bsds_id), pad_(correct_vals))
        np.save("../edge_eval_python/examples/eval-result/rced_cape/{}.npy".format(bsds_id), pad_(SC_vals))
        np.save("../edge_eval_python/examples/eval-result/rced_sc/{}.npy".format(bsds_id), pad_(cape_et_vals))
        np.save("../edge_eval_python/examples/eval-result/rced_static/{}.npy".format(bsds_id), pad_(var_et_vals))
        np.save("../edge_eval_python/examples/eval-result/rced_var/{}.npy".format(bsds_id), pad_(static_et_vals))
        
    else:
        def get_pr(A, B):
            precisions = []
            recalls = []
            for thresh in [x for x in range(0, 129, 4)]:
                mat = ConfMat(A, B, a_thresh=thresh)
                precisions.append(mat.precision())
                recalls.append(mat.recall())
            return recalls, precisions

        r_sc, p_sc = get_pr(SC_vals, correct_vals)
        r_var, p_var = get_pr(var_et_vals, correct_vals)
        r_cape, p_cape = get_pr(cape_et_vals, correct_vals)
        r_static, p_static = get_pr(static_et_vals, correct_vals)

        plt.plot(r_sc, p_sc, label="SC N=64")
        plt.plot(r_static, p_static, label="SC N=16")
        plt.plot(r_var, p_var, label="var (avg = {})".format(np.round(np.mean(var_et_Ns), 2)))
        plt.plot(r_cape, p_cape, label="cape (avg = {})".format(np.round(np.mean(cape_et_Ns), 2)))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-recall plot for bsds500 img {}".format(bsds_id))
        plt.legend()
        plt.show()