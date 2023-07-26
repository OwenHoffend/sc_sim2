from sim.RNS import *
from sim.SA import *
from experiments.early_termination_plots import *

#parr = np.array([0.5, 0.5])
#scc_vs_ne_others(parr, van_der_corput, full_width_2d, 6)
#scc_vs_ne_CAPE(parr, 6)
#scc_vs_ne_SA(0.5, 0.5, 6)

#check_ATPP(7, CAPE_sng)

sngs = [lfsr_sng, van_der_corput_sng, counter_sng, true_rand_sng, SA_sng, CAPE_sng]
tiles = [clock_division_2d_from_bs, rotation_2d_from_bs]

for sng in sngs:
    print("{}: ATPP: {}".format(sng.__name__, check_ATPP(7, sng)))
    for tile in tiles:
        print("{}: MATPP: {}".format(tile.__name__, check_MATPP(5, sng, tile)))

#check_MATPP(5, true_rand_sng)