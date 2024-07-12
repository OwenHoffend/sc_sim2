import numpy as np
from img.img_io import load_img, disp_img

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

def dataset_img_windows(img, win_sz, num=None):
    #If num is None, use the whole image, otherwise randomly sub-sample from the image
    h, w = img.shape
    print("[{},{}],".format(h, w))

    if win_sz == 1:
        ds = np.empty((h*w, 1))
        a = 0
        for i in range(h):
            for j in range(w):
                ds[a, :] = np.array([img[i, j], ])
                a+= 1

    elif win_sz == 2:
        ds = np.empty(((h-1)*(w-1), 4))
        a = 0
        for i in range(h-1):
            for j in range(w-1):
                ds[a, :] = np.array([img[i, j], img[i+1, j+1], img[i, j+1], img[i+1, j+1]])
                a+= 1
    elif win_sz == 3:
        ds = np.empty((h*w, 9))
        a = 0
        for i in range(h):
            i_ = i
            if i == 0:
                i_ = 1
            elif i == h-1:
                i_ = h-2
            for j in range(w):
                j_ = j
                if j == 0:
                    j_ = 1
                elif j == w-1:
                    j_ = w-2
                ds[a, :] = np.array([
                    img[i_-1, j_-1],
                    img[i_-1, j_],
                    img[i_-1, j_+1],
                    img[i_, j_-1],
                    img[i_, j_],
                    img[i_, j_+1],
                    img[i_+1, j_-1],
                    img[i_+1, j_],
                    img[i_+1, j_+1]
                ])
                a+= 1
    else:
        raise NotImplementedError

    if num is not None:
        row_i = np.random.choice(ds.shape[0], num, replace=False)
        ds = ds[row_i, :]
    return ds

def dataset_single_image(img_path, win_sz, num=None):
    img = load_img(img_path, gs=True, prob=True)
    return dataset_img_windows(img, win_sz, num)

def dataset_imagenet_samples(num_imgs, samps_per_img, win_sz):
    MAX_IMG_NUM = 1000
    assert num_imgs <= MAX_IMG_NUM

    img_idxs = np.random.choice(MAX_IMG_NUM, num_imgs, replace=False)
    ds = np.empty((num_imgs * samps_per_img, win_sz ** 2))

    for idx, orig_idx in enumerate(img_idxs):
        img = np.load("data/imagenet/img_{}.npy".format(orig_idx))
        ds[(samps_per_img*idx):(samps_per_img*idx+samps_per_img), :] = \
            dataset_img_windows(img, win_sz, num=samps_per_img)
    return ds

def dataset_imagenet_images(num_imgs, win_sz):
    ds = np.empty((0, 4))
    for idx in range(num_imgs):
        img = np.load("data/imagenet/img_{}.npy".format(idx))
        ds = np.concatenate((ds, dataset_img_windows(img, win_sz)))
    return ds