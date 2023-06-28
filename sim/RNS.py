import numpy as np
from pylfsr import LFSR

def lfsr(w, N):
    fpoly = LFSR().get_fpolyList(m=w)[0]
    all_zeros = np.zeros(w)
    while True:
        zero_state = np.random.randint(2, size=w) #Randomly decide where to put the init state
        if not np.all(zero_state == all_zeros):
            break
    while True:
        L = LFSR(fpoly=fpoly, initstate='random')
        if not np.all(L.state == all_zeros):
            break

    lfsr_bits = np.zeros((w, N), dtype=np.bool_)
    last_was_zero = False
    for i in range(N):
        if not last_was_zero and \
            np.all(L.state == zero_state):
                lfsr_bits[:, i] = all_zeros
                last_was_zero = True
                continue
        last_was_zero = False
        L.runKCycle(1)
        lfsr_bits[:, i] = L.state
    return lfsr_bits