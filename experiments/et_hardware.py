import numpy as np
from sim.Util import *
from sim.SCC import scc_mat
from sim.SNG import CAPE_sng
from sim.RNS import lfsr
from sim.PCC import CMP
from experiments.et_variance import get_dist
from sim.circs import robert_cross

def new_WBG(r, p):
    for i, ri in enumerate(r):
        if ri:
            return p[i]
    return False

def CAPE_basic_test():
    precision = 4
    p = np.array([0.6875,])
    Bx = parr_bin(p, precision, lsb='left')
    ctr = 0
    input_stream = np.zeros((2 ** precision), dtype=np.bool_)
    for i in range(2 ** precision):
        ctr_bin = bin_array(ctr, precision)
        input_stream[i] = new_WBG(ctr_bin, Bx[0, :])
        ctr += 1

def CAPE_corr_basic_test():
    precision = 5
    nv = 4
    p = np.random.uniform(size=nv)
    Bx = parr_bin(p, precision, lsb='right')
    ctr = 0
    input_stream = np.zeros((nv, 2 ** precision), dtype=np.bool_)
    for i in range(2 ** precision):
        ctr_bin = bin_array(ctr, precision)
        for j in range(nv):
            input_stream[j, i] = CMP(np.flip(ctr_bin), Bx[j, :])
        ctr += 1

def CAPE_test(max_precision, num_pxs, dist):
    nv = 3
    px_func = get_dist(dist, nv)
    ctr_sz = max_precision * nv
    m = 2 ** ctr_sz #2 ** ((k*s) + nc)

    def inner():
        px = px_func()
        Bx = parr_bin(px, max_precision, lsb='left')
        bs_mat = CAPE_sng(px, m, max_precision, pack=False)

        print(Bx)
        
        #test 1: normal CAPE operation without dynamic early termination
        #get the output error
        out_bs = np.bitwise_and(bs_mat[0, :], np.bitwise_and(bs_mat[1, :], bs_mat[2, :]))
        #out_bs = np.bitwise_and(bs_mat[0, :], bs_mat[1, :])
        out_prob = np.mean(out_bs)
        out_err = np.abs(out_prob - px[0] * px[1] * px[2]) 
        #out_err = np.abs(out_prob - px[0] * px[1]) 
        print("Out err: ", out_err)

        #test 2: CAPE operation with dynamic early termination
        #get the actual required precision
        req = 0
        for i in range(nv):
            for j in reversed(range(max_precision)):
                if Bx[i, j]:
                    req += j + 1
                    break
        m_et = 2 ** req
        print(m_et)

        #get the output error
        out_bs = np.bitwise_and(bs_mat[0, :m_et], np.bitwise_and(bs_mat[1, :m_et], bs_mat[2, :m_et]))
        #out_bs = np.bitwise_and(bs_mat[0, :], bs_mat[1, :])
        out_prob = np.mean(out_bs)
        out_err = np.abs(out_prob - px[0] * px[1] * px[2])
        #out_err = np.abs(out_prob - px[0] * px[1])
        print("ET out err: ", out_err)
        print('hi')

    vals = array_loop(inner, num_pxs)

def var_et_RCED(max_precision, num_pxs, dist):
    #variance-based ET done using output samples only 

    nv = 4
    px_func = get_dist(dist, nv)
    ctr_sz = max_precision
    lfsr_width = ctr_sz + 1
    m = 2 ** lfsr_width

    ets = []
    vars = []
    colors = []
    g = 0
    for _ in range(num_pxs):

        #RCED implementation
        px = px_func()
        correct = 0.5 * (np.abs(px[0] - px[1]) + np.abs(px[2] - px[3]))
        Bx = parr_bin(px, max_precision, lsb='right')
        lfsr_bits = lfsr(lfsr_width, m)
        bs_mat = np.zeros((5, m), dtype=np.bool_)
        for i in range(nv):
            p = Bx[i, :]
            for j in range(m):
                bs_mat[i, j] = CMP(lfsr_bits[:ctr_sz, j], p)
        bs_mat[4, :] = lfsr_bits[ctr_sz, :]
        
        bs_out = np.zeros((m,), dtype=np.bool_)
        for i in range(m):
            bs_out[i] = robert_cross(*list(bs_mat[:, i]))

        pz = np.mean(bs_out)
        full_length_mse = MSE(pz, correct)
        print("full length mse: ", full_length_mse)

        #dynamic ET hardware
        var = np.bitwise_and(bs_out[1:], np.bitwise_not(bs_out[:-1]))
        print("var est: " , np.mean(var))
        print("actual var: ", pz * (1-pz))

        N_et = m
        max_var = 0.01
        N_min = 24
        ell = clog2(N_min)
        cnt = N_min
        cnts = []
        for i in range(m-1):
            if var[i] and cnt < 2 ** ell - 1:
                cnt += 3
            elif not var[i] and cnt > 0:
                cnt -= 1
            if cnt == 0 and i < N_et:
                N_et = i
            cnts.append(cnt)
        print("ET at : {} out of {}".format(N_et, m))

        pz_et = np.mean(bs_out[:N_et])
        et_mse = MSE(pz_et, correct)
        print("et mse: ", et_mse)

        ets.append(N_et)
        vars.append(pz * (1-pz))
        if et_mse > max_var:
            colors.append('r')
            g+=1
        else:
            colors.append('b')

        #plt.plot(cnts)
        #plt.title("Counter value vs. N for true output variance: {} \n Terminated at: {} out of {}".format(np.round(pz * (1-pz), 3), N_et, 512))
        #plt.ylabel("Counter value")
        #plt.xlabel("N")
        #plt.show()
    #plt.scatter(vars, ets, c=colors)
    #plt.title("UNIFORM: ET N vs. var \n blue=within MSE bound, red=outside MSE bound")
    #plt.xlabel("Output variance")
    #plt.ylabel("Early termination N")
    #plt.show()
            
    plt.hist(ets, 20)
    plt.title("CENTER BETA: Hist of RCED Early Termination \n Avg length: {}, Percent >MSE thresh: {}%".format(np.mean(ets), 100* np.round(g/num_pxs, 2)))
    plt.xlabel("Early termination N")
    plt.ylabel("Frequency")
    plt.show()

    