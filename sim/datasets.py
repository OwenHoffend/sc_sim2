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

    #Code used to generate the local dataset from the online Huggingface dataset:
    #num_samps = 1000
    #ds = load_dataset("imagenet-1k")
    #img_sample = np.random.choice(ds['train'].num_rows, num_samps, replace=False)
    #imgs = ds['train'][img_sample]
    #for idx, img in enumerate(imgs['image']):
    #    print(idx)
    #    img_gs = np.array(img.convert('L')) / 256
    #    np.save("data/imagenet/img_{}".format(idx), img_gs)