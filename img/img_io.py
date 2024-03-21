import os
import numpy as np
from PIL import Image

def load_img(path, gs=False, prob=False):
    """Load an image and return it as a numpy ndarray"""
    image = Image.open(path)
    if gs: #Convert to single-channel greyscale, if desired
        image = image.convert('LA')
        width, height = image.size
        image = np.array(image)[0:height, 0:width, 0]
    else:
        image = np.array(image)
    if prob:
        return (255-image) / 256
    return image

def CIFAR_load(img_idx):
    res = cifar_unpickle("../datasets/cifar-100-python/train")
    img = np.swapaxes(res[img_idx, :].reshape(32, 32, 3, order="F"), 0, 1)
    #Image.fromarray(np.mean(img, axis=2).astype(np.uint8), "L").show()
    img_gs = np.round(np.mean(img, axis=2)) / 256 #avg all color channels
    return img_gs

def cifar_unpickle(file): #For CIFAR-10 or CIFAR-100
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data']

def img_to_bs(img_channel, bs_gen_func, bs_len=255, correlated=True, inv=False, pack=True, scale=True, lfsr_sz=None):
    """Convert a single image chnanel into an stochastic bitstream of the specified length.
    bs_gen_func is a bitstream generating function that takes in n (number of bits) and p, 
    the desired probability. If correlated is True, use the same RNG for all pixels"""
    height, width = img_channel.shape
    if pack:
        npb = np.ceil(bs_len / 8.0).astype(int) #Compute the number of packed bytes necessary to represent this bitstream
        bs = np.zeros((height, width, npb), dtype=np.uint8) #Initialize nxn array of to hold bit-packed SC bitstreams
    else:
        bs = np.zeros((height, width, bs_len), dtype=np.bool_)
    for i in range(height): #Populate the bitstream array
        for j in range(width):
            if scale:
                bs[i][j] = bs_gen_func(bs_len, float(img_channel[i][j]) / 255.0, keep_rng=correlated, inv=inv, pack=pack, lfsr_sz=lfsr_sz)
            else:
                bs[i][j] = bs_gen_func(bs_len, float(img_channel[i][j]), keep_rng=correlated, inv=inv, pack=pack, lfsr_sz=lfsr_sz)
    return bs

def bs_to_img(bs, bs_mean_func, scaling=1):
    """Convert a stochastic bitstream image back into an image"""
    height, width, npb = bs.shape
    img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height): #Populate the bitstream array
        for j in range(width):
            #Default pixel-value encoding is p * 255, might want to try others
            img[i][j] = np.rint(bs_mean_func(bs[i][j]) * 255.0 * scaling).astype(np.uint8)
    return img

def bs_to_float_img(bs, bs_mean_func):
    """Convert a stochastic bitstream image back into a (float) image"""
    height, width, npb = bs.shape
    img = np.zeros((height, width), dtype=np.float32)
    for i in range(height): #Populate the bitstream array
        for j in range(width):
            #Default pixel-value encoding is p * 255, might want to try others
            img[i][j] = bs_mean_func(bs[i][j].astype(np.float32))
    return img

def img_mse(img1, img2):
    """Compute the MSE between two images.
       Assumes the two images are the same shape"""
    h, w = img1.shape
    return np.sum(np.square(img1 - img2)) / (h * w)

def disp_img_diff(img1, img2):
    """Display the difference between img1 and img2, taken as img2 - img1"""
    diff = img2 - img1
    disp_img(diff)

def save_img(img_arr, path):
    """Save an image from a numpy ndarray at the specified path"""
    Image.fromarray(img_arr).save(path)

def disp_img(img_arr):
    """Display an image from a numpy ndarray (height, width, channels)"""
    img = Image.fromarray(img_arr)
    img.show()

def add_gauss_noise(img, sigma):
    return img + np.random.normal(0, sigma, img.shape)

def add_snp_noise(img, p, ps):
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if p > np.random.uniform():
                if ps > np.random.uniform():
                    img[i, j] = 255
                else:
                    img[i, j] = 0
    return img

def add_noise_to_dir(img_dir):
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = img_dir + img_name
        img = np.load(img_path)
        img = add_gauss_noise(img, 0.1)
        np.save(img_path, img)

def disp_all_in_dir(img_dir, scaling=255):
    img_list = os.listdir(img_dir)
    for img_name in img_list:
        img_path = img_dir + img_name
        img = np.load(img_path)
        disp_img(img*scaling)