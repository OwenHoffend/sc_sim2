from sim.RNS import *
from sim.SA import *
from sim.PTM import *
from sim.COMAX import COMAX
from sim.circs import *
from experiments.et_hardware import *

#w = get_weight_matrix_from_ptm(get_func_mat(robert_cross, 5, 1), 1, 1, 4)[:, 0]
#print(w)

#var_et_RCED(8, 250, "center_beta")

dists = ["uniform", "MNIST_beta", "center_beta"]
ns = [1, 2, 3, 4, 5, 6, 7, 8]
ws = [1, 2, 3, 4, 5, 6]

#results = np.empty((3, len(ns), len(ws)), dtype=object)
#for i, dist in enumerate(dists):
#    for j, n in enumerate(ns):
#        for k, w in enumerate(ws):
#            if n * w >= 32:
#                continue
#            results[i, j, k] = CAPE_based_ET_stats(n, w, dist, 10000)
#np.save("cape_ET.npy", results)

results = np.load("cape_ET.npy", allow_pickle=True)

for d in range(3):
    for i in range(len(ws)):
        plt.plot(ns, results[d, :, i], label="{}-bits of precision".format(i+1))
    plt.title("Relative bitstream length, {} \n (Lower is better)".format(dists[d]))
    plt.xlabel("Number of inputs (n)")
    plt.ylabel("Relative bitstream length")
    plt.legend()
    plt.show()