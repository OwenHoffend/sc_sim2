from img.img_io import *
from experiments.et_hardware import RCED_et_kernel

def et_RCED_on_CIFAR100(num_imgs):
    res = cifar_unpickle("../datasets/cifar-100-python/train")
    h, w = 32, 32
    max_precision = 8
    staticN = 64
    max_var = 0.01
    #for img_idx in [1, 3, 5, 6]:
    #    img = np.swapaxes(res[img_idx, :].reshape(32, 32, 3, order="F"), 0, 1)
    #    img_gs = np.round(np.mean(img, axis=2)) / 256 #avg all color channels
    #    Image.fromarray(np.mean(img, axis=2).astype(np.uint8), "L").show()

    for img_idx in [1, 3, 5, 6]:
        img = np.swapaxes(res[img_idx, :].reshape(32, 32, 3, order="F"), 0, 1)
        img_gs = np.round(np.mean(img, axis=2)) / 256 #avg all color channels
        correct_img = np.zeros((h-1, w-1))
        sc_full_img = np.zeros((h-1, w-1))
        var_et_img = np.zeros((h-1, w-1))
        cape_et_img = np.zeros((h-1, w-1))
        avg_N_var = 0.0
        avg_N_CAPE = 0.0
        for i in range(h-1):
            print(i)
            for j in range(w-1):
                px = np.array([img_gs[i, j], img_gs[i+1, j+1], img_gs[i, j+1], img_gs[i+1, j]])
                correct, pz_full, pz_et_var, N_et_var, pz_et_CAPE, N_et_CAPE = \
                    RCED_et_kernel(px, max_precision, max_var, staticN=staticN)
                correct_img[i, j] = correct
                sc_full_img[i, j] = pz_full
                var_et_img[i, j] = pz_et_var
                cape_et_img[i, j] = pz_et_CAPE
                avg_N_var += N_et_var
                avg_N_CAPE += N_et_CAPE
        avg_N_var /= 32 * 32
        avg_N_CAPE /= 32 * 32
        print("avg N var: ", avg_N_var)
        print("avg N CAPE: ", avg_N_CAPE)
        print("MSE SC full: ", img_mse(correct_img, sc_full_img))
        print("MSE var: ", img_mse(correct_img, var_et_img))
        print("MSE CAPE: ", img_mse(correct_img, cape_et_img))
        Image.fromarray(2 * np.round(correct_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_correct.png".format(img_idx))
        Image.fromarray(2 * np.round(sc_full_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_sc_full.png".format(img_idx))
        Image.fromarray(2 * np.round(var_et_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_var_et.png".format(img_idx))
        Image.fromarray(2 * np.round(cape_et_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_CAPE_et.png".format(img_idx))
