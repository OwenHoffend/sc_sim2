from sim.RNS import *
from sim.SA import *
from sim.PTM import *
from sim.COMAX import COMAX
from sim.circs import *
from sim.ReSC import *
from img.img_io import *
from sim.ATPP import *
from experiments.ret_vs_set import *
from experiments.et_hardware import *
from experiments.et_on_images import *
from experiments.early_termination_plots import *

if __name__ == "__main__":
    #et_RCED_on_imgs(load_img("./data/cameraman.png", gs=True, prob=True), "cameraman_inv")
    
    #w = 5
    #xs = [x/(2**w) for x in range(2 ** w)]
    #avg_used_prec(xs, 5)

    #heatmap_actual_precision_use(5)

    img = load_img("./data/cameraman.png", gs=True)
    prec_util_of_img(img)

    #p = 0.33
    #lfsr_bs = lfsr_sng(np.array([p, ]), 256, 8, pack=False)
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
    #    [lfsr_bs, cape_var],
    #    ["lfsr", "cape_with_var_et"],
    #    cape_bs.size, N_et, p
    #)