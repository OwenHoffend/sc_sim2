from sim.RNS import *
from sim.SA import *
from sim.PTM import *
from sim.COMAX import COMAX
from sim.circs import *
from sim.ReSC import *
from img.img_io import *
from sim.ATPP import *
from experiments.early_termination.precision_analysis import *
from experiments.early_termination.et_hardware import *
from experiments.early_termination.et_on_images import *
from experiments.early_termination.early_termination_plots import *
from experiments.early_termination.et_gamma import *
from experiments.early_termination.et_AND import *

if __name__ == "__main__":
    x_squared_et(8)

    #et_RCED_on_imgs(load_img("./data/cameraman.png", gs=True, prob=True), "cameraman_inv")

    #gamma_correction(num_samples=25, w=8, Nmax=2**8, max_var=0.001)
    
    #w = 5
    #xs = [x/(2**w) for x in range(2 ** w)]
    #avg_used_prec(xs, 5)

    #heatmap_actual_precision_use(5)

    #actual = 2 ** np.array(list(reversed(range(1, 9))))

    #title = "cameraman"
    #img = load_img("./data/{}.png".format(title), gs=True)
    #ssims = prec_util_of_img(img, title)
    #plt.plot(ssims / actual, label="cameraman, RET")

    #title = "lena"
    #img = load_img("./data/{}.png".format(title), gs=True)
    #ssims = prec_util_of_img(img, title)
    #plt.plot(ssims / actual, label="lena RET")

    #title = "house"
    #img = load_img("./data/{}.png".format(title), gs=True)
    #ssims = prec_util_of_img(img, title)
    #plt.plot(ssims / actual, label="house REG")

    #plt.title("Savings ratio vs. # of bits truncated")
    #plt.ylabel("Savings ratio")
    #plt.xlabel("# of bits truncated")
    #plt.legend()
    #plt.show()

    #print(prec_util(0b11111, 5))

    #p = 0.5
    #lfsr_bs = van_der_corput_sng(np.array([p, ]), 256, 8, pack=False)

    #p2 = 0.5 + 1/128
    #lfsr_bs_2 = van_der_corput_sng(np.array([p2, ]), 256, 8, pack=False)

    #p3 = 0.125
    #lfsr_bs_3 = van_der_corput_sng(np.array([p3, ]), 256, 8, pack=False)

    #van_bs = van_der_corput_sng(np.array([p, ]), 256, 8, pack=False)
    #cape_bs = CAPE_sng(np.array([p, ]), 8, [1, ], et=True)
    #cape_bs_full = CAPE_sng(np.array([p, ]), 8, [1, ])
    
    #N_et, cnts = var_et(lfsr_bs, 0.01)
    #lfsr_var_et = lfsr_bs[:N_et]

    #N_et, cnts = var_et(cape_bs_full, 0.01)
    #cape_var = cape_bs_full[:N_et]

    #plt.plot(cnts)
    #plt.show()

    #partial_bitstream_value_plot(
    #    [lfsr_bs, lfsr_bs_2, lfsr_bs_3],
    #    [p, p2, p3]
    #)