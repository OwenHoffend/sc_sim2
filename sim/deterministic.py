import numpy as np
from sim.Util import bit_vec_arr_to_int

"""
Deterministic methods of cycling through all possible RNS states for multiple bitstreams
Techniques from "A Deterministic Approach to Stochastic Computation" - Devon Jenson & Marc Riedel, 2016
"""

def primes_from_2_to(n):
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n//3 + (n%6==2), dtype=np.bool_)
    sieve[0] = False
    for i in range(int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[      ((k*k)//3)      ::2*k] = False
            sieve[(k*k+4*k-2*k*(i&1))//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]

def relative_primes():
    pass

def full_width_2d(rns, w, N):
    full_N = N ** 2
    rand = rns(2*w, full_N)
    x_rns = bit_vec_arr_to_int(rand[:w])
    y_rns = bit_vec_arr_to_int(rand[w:])
    return x_rns, y_rns

def rotation_2d(rns, w, N):
    rand_x = bit_vec_arr_to_int(rns(w, N))
    rand_y = bit_vec_arr_to_int(rns(w, N))
    x_rns = np.empty((0, ), dtype=np.int32)
    y_rns = np.empty((0, ), dtype=np.int32)
    for _ in range(N):
        x_rns = np.concatenate((x_rns, rand_x))
        y_rns = np.concatenate((y_rns, rand_y))
        rand_y = np.roll(rand_y, 1)
    return x_rns, y_rns

def clock_division_2d(rns, w, N):
    rand_x = bit_vec_arr_to_int(rns(w, N))
    rand_y = bit_vec_arr_to_int(rns(w, N))
    x_rns = np.empty((0, ), dtype=np.int32)
    y_rns = np.empty((0, ), dtype=np.int32)
    for i in range(N):
        x_rns = np.concatenate((x_rns, rand_x))
        y_rns = np.concatenate((y_rns, np.array(N*[rand_y[i]])))
    return x_rns, y_rns