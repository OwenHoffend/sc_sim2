import numpy as np
import os
from img.img_io import *

class Dataset:
    """Wrapper class for datasets input to stochastic circuits in the simulator"""
    def __init__(self, ds):
        self.ds = ds
        self.num = ds.shape[0]
        self.n = ds.shape[1]

    #Merge two datasets together. By default, keeps the number of samples the same,
    #but adds data for additional circuit inputs.
    #This was made because I wanted to add correlated 0.5-valued constants to certain datasets
    #Eg: ds = ds.merge(dataset_all_same(ds.num, 1, 0.5))
    def merge(self, ds2, axis=1):
        return Dataset(np.concatenate((self.ds, ds2.ds), axis=axis))

    def __iter__(self):
        for i in range(self.num):
            yield self.ds[i, :]

class ImageDataset(Dataset):
    def __init__(self, imgs: list | np.ndarray, win_sz, windows_per_img=None):
        """imgs is a list of image 2D np.ndarrays holding grayscale images probabilities"""
        self.win_sz = win_sz

        if isinstance(imgs, list):
            self.imgs = imgs
        else:
            self.imgs = [imgs, ]

        self.shapes = []
        for img in self.imgs:
            self.shapes.append(img.shape)

        #TODO: implement file streaming to handle more images without eating up memory
        ds = np.empty((0, win_sz ** 2))
        for img in self.imgs:
            ds = np.concatenate((ds, get_img_windows(img, win_sz, num=windows_per_img)))

        super().__init__(ds)

    def disp_img(self, idx, scale=True, colorbar=False):
        if scale:
            scale_factor = 255
        else:
            scale_factor = 1
        disp_img(scale_factor * self.imgs[idx], colorbar=colorbar)

    def disp_output_img(self, img, idx, scale=True, colorbar=False):
        shape = self.shapes[idx]
        shape = (shape[0] - (self.win_sz - 1), shape[1] - (self.win_sz - 1))
        if scale:
            scale_factor = 255
        else:
            scale_factor = 1
        disp_img(scale_factor * img.reshape(shape), colorbar=colorbar)

    @staticmethod
    def imgs_from_img_folder(dir, mode='random', num_imgs=10, idxs=None) -> list[np.ndarray]:
        """Expects a directory containing a bunch of .npy files, where each .npy file is an image"""

        #Operating modes:
        #random: randomly pick num images from the image folder
        #Ind list: Provide a list of specific images we want to use, by filename
        #seq: use up to the first num images
        
        files = os.listdir(dir)
        num_files = len(files)

        if mode == "random":
            assert num_imgs is not None
            if num_imgs >= num_files:
                raise ValueError("Number of requested images is greater than number in directory")
            img_idxs = np.random.choice(num_files, num_imgs, replace=False)
            return [np.load(os.path.join(dir, files[idx])) for idx in img_idxs] 
        
        elif mode == "list":
            assert idxs is not None
            imgs = []
            for idx in idxs:
                if idx >= num_files:
                    raise ValueError("Image number {} does not exist".format(idx))
                imgs.append(np.load(os.path.join(dir, files[idx])))
            return imgs
        
        elif mode == "seq":
            assert num_imgs is not None
            if num_imgs > num_files:
                raise ValueError("Number of requested images is greater than number in directory")
            return [np.load(os.path.join(dir, files[idx])) for idx in range(num_imgs)]
        else:
            raise NotImplementedError("Image folder mode not implemented")

def dataset_imagenet(win_sz, windows_per_img=None, mode='random', num_imgs=10, idxs=None) -> ImageDataset:
    imgs = ImageDataset.imgs_from_img_folder("data/imagenet/", mode=mode, num_imgs=num_imgs, idxs=idxs)
    return ImageDataset(imgs, win_sz, windows_per_img=windows_per_img)

def dataset_mnist(win_sz, windows_per_img=None, mode='random', num_imgs=10, idxs=None) -> ImageDataset:
    imgs = ImageDataset.imgs_from_img_folder("data/mnist/", mode=mode, num_imgs=num_imgs, idxs=idxs)
    return ImageDataset(imgs, win_sz, windows_per_img=windows_per_img)

def dataset_cameraman(win_sz):
    imgs = ImageDataset.imgs_from_img_folder("data/cameraman", mode='seq', num_imgs=1)
    return ImageDataset(imgs, win_sz)

def dataset_uniform(num, n) -> Dataset:
    return Dataset(np.random.uniform(size=(num, n)))

def dataset_mnist_beta(num, n) -> Dataset:
    return Dataset(np.random.beta(0.0362, 0.1817, size=(num, n)))

def dataset_center_beta(num, n) -> Dataset:
    return Dataset(np.random.beta(3, 3, size=(num, n)))

def dataset_all_same(num, n, val) -> Dataset:
    return Dataset(np.full((num, n), val))

def dataset_discrete(num, n, vals, probs) -> Dataset:
    return Dataset(np.random.choice(vals, size=(num, n), p=probs))

def dataset_sweep_1d(num) -> Dataset:
    return Dataset(np.expand_dims(np.linspace(0, 1, num), axis=1))

def dataset_single(pxs):
    return Dataset(np.array([pxs]))

def dataset_sweep_2d(nx, ny) -> Dataset:
    #This code might be extendable to a general sweep of n variables
    grid = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    return Dataset(np.vstack((grid[0].flatten(), grid[1].flatten())).T)