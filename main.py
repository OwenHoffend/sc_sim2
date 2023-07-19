from sim.RNS import *
from sim.streaming_accuracy import *
from experiments.early_termination_plots import *

bs = streaming_accurate_SNG(0.1, 16)
print(np.mean(bs) / 256)
print(SA(bs))