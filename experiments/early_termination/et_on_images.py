from img.img_io import *
import matplotlib.pyplot as plt
from experiments.early_termination.et_RCED import RCED_et_kernel
from img.img_quality import ssim

def prec_util(val, w):
    mask = 0b1
    for i in reversed(range(w)):
        if not ~val & mask:
            return i+1
        mask <<= 1
    return 0

def prec_util_of_img(img, title):
    mask = 0b11111111
    h, w = img.shape
    ssims = []
    for i in range(8):
        mask >>= i
        mask <<= i
        img_trunc = img & mask

        prec_img = np.empty((h, w), dtype=int)
        for y in range(h):
            for x in range(w):
                prec_img[y, x] = prec_util(img_trunc[y, x], 8)

        #disp_img(img_trunc)
        #print("Trunc to bits: ", 8-i)
        #print("SSIM: ", ssim(img, img_trunc))
        ssims.append(np.mean(2 ** prec_img))
        #ssims.append(ssim(img, img_trunc))
        unq, counts = np.unique(prec_img, return_counts=True)
        plt.plot(unq, counts, marker='o', label=8-i)
    plt.xlabel("Used bits of precision")
    plt.ylabel("Frequency")
    plt.title("Image pixel precision frequencies. Img: {}".format(title))
    plt.legend(title="Trunc. width")
    plt.show()
    return ssims

def et_RCED_on_imgs(img, name):
    h, w = img.shape
    max_precision = 6
    max_var = 0.01

    for staticN in [22, 27, 28, 31]:
        correct_img = np.zeros((h-1, w-1))
        sc_full_img = np.zeros((h-1, w-1))
        var_et_img = np.zeros((h-1, w-1))
        cape_et_img = np.zeros((h-1, w-1))
        N_var_img = np.zeros((h-1, w-1))
        N_CAPE_img = np.zeros((h-1, w-1))
        avg_N_var = 0.0
        avg_N_CAPE = 0.0

        print(staticN)
        for i in range(h-1):
            #start_time = time.time()
            for j in range(w-1):
                px = np.array([img[i, j], img[i+1, j+1], img[i, j+1], img[i+1, j+1]])
                #correct, pz_full, pz_et_var, N_et_var, pz_et_CAPE, N_et_CAPE = \
                correct, pz_full = \
                    RCED_et_kernel(px, max_precision, max_var, staticN=staticN)
                correct_img[i, j] = correct
                sc_full_img[i, j] = pz_full
                #var_et_img[i, j] = pz_et_var
                #cape_et_img[i, j] = pz_et_CAPE
                #N_var_img[i, j] = N_et_var
                #N_CAPE_img[i, j] = N_et_CAPE
                #avg_N_var += N_et_var
                #avg_N_CAPE += N_et_CAPE
            #print("--- Full time: %s seconds ---" % (time.time() - start_time))

    #plt.imshow(N_var_img, cmap='hot', interpolation='nearest')
    #plt.colorbar()
    #plt.title("N Var")
    #plt.show()

    #plt.imshow(N_CAPE_img, cmap='hot', interpolation='nearest')
    #plt.colorbar()
    #plt.title("N CAPE")
    #plt.show()

        avg_N_var /= h * w
        avg_N_CAPE /= h * w
        print("max prec: ", max_precision)
        print("avg N var: ", avg_N_var)
        print("avg N CAPE: ", avg_N_CAPE)
        print("MSE SC full: ", img_mse(correct_img, sc_full_img))
        print("MSE var: ", img_mse(correct_img, var_et_img))
        print("MSE CAPE: ", img_mse(correct_img, cape_et_img))
    #Image.fromarray(2 * np.round(correct_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_correct.png".format(name))
    #Image.fromarray(2 * np.round(sc_full_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_sc_full.png".format(name))
    #Image.fromarray(2 * np.round(var_et_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_var_et.png".format(name))
    #Image.fromarray(2 * np.round(cape_et_img * 256).astype(np.uint8), "L").save("./results/ET_RCED/{}_CAPE_et.png".format(name))
