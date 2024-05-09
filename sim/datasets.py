import numpy as np

def dataset_uniform(num, n):
    return np.random.uniform(size=(num, n))

def dataset_mnist_beta(num, n):
    return np.random.beta(0.0362, 0.1817, size=(num, n))

def dataset_center_beta(num, n):
    return np.random.beta(3, 3, size=(num, n))

def dataset_all_same(num, n, val):
    return np.full((num, n), val)