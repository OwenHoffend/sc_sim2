import numpy as np
from img.img_io import load_img

def dataset_uniform(num, n):
    return np.random.uniform(size=(num, n))

def dataset_mnist_beta(num, n):
    return np.random.beta(0.0362, 0.1817, size=(num, n))

def dataset_center_beta(num, n):
    return np.random.beta(3, 3, size=(num, n))

def dataset_all_same(num, n, val):
    return np.full((num, n), val)

def dataset_discrete(num, n, vals, probs):
    return np.random.choice(vals, size=(num, n), p=probs)

def dataset_sweep_1d(num):
    return np.expand_dims(np.linspace(0, 1, num), axis=1)

def dataset_img_windows(img_path, img_sz, num=None):
    img = load_img(img_path, gs=True, prob=True)
    h, w = img.shape

    if img_sz == 1:
        ds = np.empty((h*w, 1))
        for i in range(h):
            for j in range(w):
                ds[i*(h-1) + j, :] = np.array([img[i, j], ])
    elif img_sz == 2:
        ds = np.empty(((h-1)*(w-1), 4))
        for i in range(h-1):
            for j in range(w-1):
                ds[i*(h-1) + j, :] = np.array([img[i, j], img[i+1, j+1], img[i, j+1], img[i+1, j+1]])
    else:
        raise NotImplementedError

    if num is not None:
        row_i = np.random.choice(ds.shape[0], num, replace=False)
        ds = ds[row_i, :]
    return ds