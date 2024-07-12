from sim.RNS import *
from sim.SA import *
from sim.PTM import *
from sim.COMAX import COMAX
from sim.circs import *
from sim.ReSC import *
from img.img_io import *
from sim.ATPP import *
from multiprocessing import Pool
from sim.datasets import *
from experiments.early_termination.precision_analysis import *
from experiments.early_termination.et_hardware import *
from experiments.early_termination.early_termination_plots import *
from experiments.early_termination.et_ed import *
from experiments.early_termination.SET import *

def test(i):
    print(i)
    img = np.load("data/imagenet/img_{}.npy".format(i))
    ds = dataset_img_windows(img, 2)
    circ = C_RCED()
    return ET_sim(ds, circ, 0.02, 0.03, SET_override=(256, 128, 64), j=i)

def test2(i):
    print(i)
    img = np.load("data/imagenet/img_{}.npy".format(i))
    ds = dataset_img_windows(img, 2)
    circ = C_RCED()
    w, Nmax, Nset = ideal_SET(ds, circ, 0.01, 0.05)
    return Nset

if __name__ == "__main__":
    #fig_1(100, Nmax=64)

    #The other side of this though is that it's possible for the correct value to be very
    #small, to the point where 0 is actually the correct approximation
    #ds1 = dataset_mnist_beta(1000, 1)
    #ds2 = dataset_center_beta(1000, 1)
    #ds3 = dataset_uniform(1000, 2)
    #circ = C_WIRE()

    #ds4 = dataset_single_image("./data/cameraman.png", 2, 1000)
    #ds5 = dataset_single_image("./data/lena.png", 2, 1000)
    #ds6 = dataset_single_image("./data/house.png", 2, 1000)
    #ds7 = dataset_single_image("./data/mnist.png", 2, 1000)

    #ds = dataset_imagenet_images(10, 2)
    #circ = C_RCED()
    #ET_sim(ds, circ, 0.02, 0.03)

    #with Pool(10) as p:
    #    result = p.map(test, [x for x in range(10)])
    #print(result)
    #pass

    i = 2
    img = np.load("data/imagenet/img_{}.npy".format(i))
    #ds = dataset_img_windows(img, 2, num=10000)
    ds = dataset_uniform(100, 2)
    circ = C_AND_N(2)
    ET_sim(ds, circ, 0.01, 0.05)

    #scatter_ET_results(0.03, 0.05)