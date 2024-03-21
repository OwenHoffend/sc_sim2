from img.img_io import *
import matplotlib.pyplot as plt
from experiments.et_hardware import RCED_et_kernel
from multiprocessing import Pool
import time

#def pkernel(i):
#    correct_vec = np.zeros((w-1,))
#    sc_full_vec = np.zeros((w-1,))
#    var_et_vec = np.zeros((w-1,))
#    cape_et_vec = np.zeros((w-1,))
#    avg_N_var_ = 0.0
#    avg_N_CAPE_ = 0.0
#    for j in range(w-1):
#        px = np.array([img_gs[i, j], img_gs[i+1, j+1], img_gs[i, j+1], img_gs[i+1, j]])
#        correct, pz_full, pz_et_var, N_et_var, pz_et_CAPE, N_et_CAPE = \
#            RCED_et_kernel(px, max_precision, max_var, staticN=staticN)
#        correct_vec[j] = correct
#        sc_full_vec[j] = pz_full
#        var_et_vec[j] = pz_et_var
#        cape_et_vec[j] = pz_et_CAPE
#        avg_N_var_ += N_et_var
#        avg_N_CAPE_ += N_et_CAPE
#
#    return correct_vec, sc_full_vec, var_et_vec, cape_et_vec, avg_N_var_, avg_N_CAPE_

def et_RCED_on_imgs(img, name):
    h, w = img.shape
    max_precision = 6
    max_var = 0.01

    correct_img = np.zeros((h-1, w-1))
    sc_full_img = np.zeros((h-1, w-1))
    var_et_img = np.zeros((h-1, w-1))
    cape_et_img = np.zeros((h-1, w-1))
    N_var_img = np.zeros((h-1, w-1))
    N_CAPE_img = np.zeros((h-1, w-1))
    avg_N_var = 0.0
    avg_N_CAPE = 0.0

    #with Pool(20) as p:
    #    rows = p.map(pkernel, list(range(h-1)))
    #    pass

    for i in range(h-1):
        print(i)
        start_time = time.time()
        for j in range(w-1):
            px = np.array([img[i, j], img[i+1, j+1], img[i, j+1], img[i+1, j]])
            correct, pz_full, pz_et_var, N_et_var, pz_et_CAPE, N_et_CAPE = \
                RCED_et_kernel(px, max_precision, max_var)
            correct_img[i, j] = correct
            sc_full_img[i, j] = pz_full
            var_et_img[i, j] = pz_et_var
            cape_et_img[i, j] = pz_et_CAPE
            N_var_img[i, j] = N_et_var
            N_CAPE_img[i, j] = N_et_CAPE
            avg_N_var += N_et_var
            avg_N_CAPE += N_et_CAPE
        print("--- Full time: %s seconds ---" % (time.time() - start_time))

    plt.imshow(N_var_img, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("N Var")
    plt.show()

    plt.imshow(N_CAPE_img, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("N CAPE")
    plt.show()

    avg_N_var /= h * w
    avg_N_CAPE /= h * w
    print("avg N var: ", avg_N_var)
    print("avg N CAPE: ", avg_N_CAPE)
    print("MSE SC full: ", img_mse(correct_img, sc_full_img))
    print("MSE var: ", img_mse(correct_img, var_et_img))
    print("MSE CAPE: ", img_mse(correct_img, cape_et_img))
    Image.fromarray(2 * np.round(correct_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_correct.png".format(name))
    Image.fromarray(2 * np.round(sc_full_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_sc_full.png".format(name))
    Image.fromarray(2 * np.round(var_et_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_var_et.png".format(name))
    Image.fromarray(2 * np.round(cape_et_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_CAPE_et.png".format(name))
