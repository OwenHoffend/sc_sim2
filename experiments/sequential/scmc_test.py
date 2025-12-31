import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sim.circs.SCMCs import C_FSM_SYNC
from sim.circs.circs import C_WIRE
from sim.SNG import LFSR_SNG, RAND_SNG, nonint_scc
from sim.SCC import scc_mat
from sim.PTV import get_vin_nonint_pair, get_C_from_v
from symb_analysis.CAP import get_sym_ptv
from symb_analysis.seq_CAP import get_steady_state
from sim.datasets import beta_pdf
from sim.Util import BMSE
from sim.circs.SCMCs import fsm_reco_abdellatef

def sim_fsm_sync():
    """Simulation of FSM synchronizers with respect to depth d"""

    num_sccs = 100
    sccs = np.linspace(-1, 1, num_sccs)
    px = 0.5
    py = 0.5
    w = 12
    N = 2048
    d_abdellatef = 2
    for d in range(1, 2):
        print("Depth={}".format(d))
        input_sccs = []
        output_sccs = []
        output_sccs_abdellatef = []
        sync = C_FSM_SYNC(d)
        for scc in sccs:
            cin_uncorr = np.eye(2)
            
            if scc < 0:
                cin_corr = -np.ones((2, 2)) + 2 * np.eye(2)
            else:
                cin_corr = np.ones((2, 2))
            #sng = LFSR_SNG(w, C_WIRE(2, cin_uncorr))
            #sng2 = LFSR_SNG(w, C_WIRE(2, cin_corr))
            sng = RAND_SNG(w, C_WIRE(2, cin_uncorr))
            sng2 = RAND_SNG(w, C_WIRE(2, cin_corr))
            bs_mat = sng.run(np.array([px, py]), N)
            bs_mat2 = sng2.run(np.array([px, py]), N)
            bs_mat_out = nonint_scc(bs_mat, bs_mat2, scc)
            input_scc = scc_mat(bs_mat_out)[0, 1]
            input_sccs.append(input_scc)
            bs_mat_sync = sync.run(bs_mat_out)
            output_scc = scc_mat(bs_mat_sync)[0, 1]
            output_sccs.append(output_scc)

            if d == 1:
                #Also simulate the Abdellatef design using depth = 3 from their paper
                bs_mat_out_abdellatef = fsm_reco_abdellatef(bs_mat_out[0, :], bs_mat_out[1, :], d_abdellatef, d_abdellatef)
                output_scc_abdellatef = scc_mat(bs_mat_out_abdellatef)[0, 1]
                output_sccs_abdellatef.append(output_scc_abdellatef)
        if d == 1:      
            plt.scatter(sccs, input_sccs, label="Input SCC")
            plt.scatter(sccs, output_sccs_abdellatef, label="Output SCC Abdellatef depth={}".format(d_abdellatef))

        plt.scatter(sccs, output_sccs, label="Output SCC depth={}".format(d))
    plt.title("Input SCC vs Output SCC")
    plt.xlabel("Test SCC")
    plt.ylabel("Output SCC")
    plt.legend()
    plt.show()

def sim_fsm_sync_px_sweep():
    """Simulation of FSM synchronizers but sweep over a range of px and py values"""

    num_sccs = 100
    sccs = np.linspace(-1, 1, num_sccs)
    w = 10
    N = 1024

    num_pvals = 15
    output_sccs1 = np.zeros((num_pvals ** 2, num_sccs))
    output_sccs2 = np.zeros((num_pvals ** 2, num_sccs))
    output_sccs_abdellatef_3 = np.zeros((num_pvals ** 2, num_sccs))
    output_sccs_abdellatef_4 = np.zeros((num_pvals ** 2, num_sccs))
    output_sccs_abdellatef_impr1 = np.zeros((num_pvals ** 2, num_sccs))

    idx = 0
    for px in np.linspace(0, 1, num_pvals):
        print("px={}".format(px))
        for py in np.linspace(0, 1, num_pvals):
            sync1 = C_FSM_SYNC(1)
            sync2 = C_FSM_SYNC(3)
            for scc_idx, scc in enumerate(sccs):
                cin_uncorr = np.eye(2)
                
                if scc < 0:
                    cin_corr = -np.ones((2, 2)) + 2 * np.eye(2)
                else:
                    cin_corr = np.ones((2, 2))
                #sng = LFSR_SNG(w, C_WIRE(2, cin_uncorr))
                #sng2 = LFSR_SNG(w, C_WIRE(2, cin_corr))
                sng = RAND_SNG(w, C_WIRE(2, cin_uncorr))
                sng2 = RAND_SNG(w, C_WIRE(2, cin_corr))
                bs_mat = sng.run(np.array([px, py]), N)
                bs_mat2 = sng2.run(np.array([px, py]), N)
                bs_mat_out = nonint_scc(bs_mat, bs_mat2, scc)
                bs_mat_sync1 = sync1.run(bs_mat_out)
                bs_mat_sync2 = sync2.run(bs_mat_out)
                output_scc1 = scc_mat(bs_mat_sync1)[0, 1]
                output_scc2 = scc_mat(bs_mat_sync2)[0, 1]
                output_sccs1[idx, scc_idx] = output_scc1
                output_sccs2[idx, scc_idx] = output_scc2

                #Also simulate the Abdellatef design using depth = 3 and depth = 4 from their paper
                bs_mat_out_abdellatef = fsm_reco_abdellatef(bs_mat_out[0, :], bs_mat_out[1, :], 3, 3)
                output_scc_abdellatef_3 = scc_mat(bs_mat_out_abdellatef)[0, 1]
                output_sccs_abdellatef_3[idx, scc_idx] = output_scc_abdellatef_3

                bs_mat_out_abdellatef = fsm_reco_abdellatef(bs_mat_out[0, :], bs_mat_out[1, :], 4, 4)
                output_scc_abdellatef_4 = scc_mat(bs_mat_out_abdellatef)[0, 1]
                output_sccs_abdellatef_4[idx, scc_idx] = output_scc_abdellatef_4

                bs_mat_out_abdellatef_impr1 = fsm_reco_abdellatef(bs_mat_out[0, :], bs_mat_out[1, :], 3, 3, impr1=True)
                output_scc_abdellatef_impr1 = scc_mat(bs_mat_out_abdellatef_impr1)[0, 1]
                output_sccs_abdellatef_impr1[idx, scc_idx] = output_scc_abdellatef_impr1
            idx += 1

    plt.scatter(sccs, output_sccs_abdellatef_3.mean(axis=0), label="Output SCC Abdellatef depth=3")
    plt.scatter(sccs, output_sccs_abdellatef_4.mean(axis=0), label="Output SCC Abdellatef depth=4")
    plt.scatter(sccs, output_sccs1.mean(axis=0), color="gold", label="Output SCC V. Lee depth=1")
    plt.scatter(sccs, output_sccs_abdellatef_impr1.mean(axis=0), label="Output SCC Abdellatef depth=3 (impr1)")
    plt.scatter(sccs, output_sccs2.mean(axis=0), color="red", label="Output SCC V. Lee depth=3")
    plt.title("Input SCC vs Output SCC")
    plt.xlabel("Test SCC")
    plt.ylabel("Output SCC")
    plt.ylim(0.8, 1)
    plt.legend()
    plt.show()

def test_CAP_fsm_sync():
    num_sccs = 100
    sccs = np.linspace(-1, 1, num_sccs)
    px = 0.9
    py = 0.1
    for d in range(1, 4):

        #NOTE: A lot of this code can be moved to the final seq-CAP implementation
        print("Depth={}".format(d))
        output_sccs = []
        circ = C_FSM_SYNC(d)
        dv = np.array(sp.symbols('v0 v1 v2 v3', real=True, nonneg=True))
        T = circ.get_T(dv)
        pi = get_steady_state(T, vars=dv)
        ptm = circ.get_PTM(pi)
        for scc in sccs:

            #Switching to numeric evaluation here because it's more efficient
            vin = get_vin_nonint_pair(scc, px, py)
            for idx, v in enumerate(dv):
                ptm = ptm.subs(v, vin[idx])
            scc_out = get_C_from_v(ptm.T @ vin)[0, 1]
            output_sccs.append(scc_out)
        if d == 1:
            plt.plot(sccs, sccs, label="Input SCC")
        plt.plot(sccs, output_sccs, label="Output SCC depth={}".format(d))
        print(output_sccs)
    plt.title("Input SCC vs Output SCC")
    plt.xlabel("Test SCC")
    plt.ylabel("Output SCC")
    plt.legend()
    plt.show()

def synchronizer_symbolic_curves():
    #This function produced the plots in the 10/4/2025 report for the behavior of an AND gate min(X, 0.5) function

    for d in range(1, 4):
        circ = C_FSM_SYNC(d)
        dv = np.array(sp.symbols('v0 v1 v2 v3', real=True, nonneg=True))
        T = circ.get_T(dv)
        pi = get_steady_state(T, vars=dv)
        ptm = circ.get_PTM(pi)
        vin = get_sym_ptv(np.array([[1, 0], [0, 1]]))
        for idx, v in enumerate(dv):
            ptm = ptm.subs(v, vin[idx])
        vout = sp.nsimplify(sp.Matrix(ptm.T @ vin))
        x1, x2 = sp.symbols('x1 x2', real=True, nonneg=True)
        print(sp.latex(sp.nsimplify(sp.simplify(vout[3].subs(x2, 0.5)))))
        
        # Substitute x2=0.5 in vout[3] and plot the result with respect to x1
        vout3_x2_05 = vout[3].subs(x2, 0.5)
        x1_vals = np.linspace(0, 1, 200)
        vout3_vals = [float(vout3_x2_05.subs(x1, val)) for val in x1_vals]
        plt.plot(x1_vals, vout3_vals, label="FSM, depth={}".format(d))
    plt.plot(np.linspace(0, 1, 200), np.minimum(np.linspace(0, 1, 200), 0.5), label='ideal')
    plt.xlabel("x1")
    plt.ylabel("vout[3] (x2=0.5)")
    plt.title("vout[3] vs x1 with x2=0.5")
    plt.grid(True)
    plt.legend()
    plt.show()

def synchronizer_symbolic_error():
    #Very similar to the synchonizer_symbolic_curves function but calculates Bayesian error instead

    x1_vals = np.linspace(0, 1, 200)
    ideal_vals = np.minimum(x1_vals, 0.5)
    x1, x2 = sp.symbols('x1 x2', real=True, nonneg=True)

    #Get the beta distribution pdf for the three different distributions used in Tim's paper
    center_beta_pdf = beta_pdf(x1, 3, 3)
    uniform_pdf = beta_pdf(x1, 1, 1)
    left_bias_pdf = beta_pdf(x1, 3, 8)
    right_bias_pdf = beta_pdf(x1, 8, 3)

    # Plot the beta distribution pdfs
    #plt.figure()
    #plt.plot(x1_vals, [float(center_beta_pdf.subs(x1, val)) for val in x1_vals], label='Beta(3,3) Centered')
    #plt.plot(x1_vals, [float(uniform_pdf.subs(x1, val)) for val in x1_vals], label='Beta(1,1) Uniform')
    #plt.plot(x1_vals, [float(left_bias_pdf.subs(x1, val)) for val in x1_vals], label='Beta(3,8) Left-biased')
    #plt.plot(x1_vals, [float(right_bias_pdf.subs(x1, val)) for val in x1_vals], label='Beta(8,3) Right-biased')
    #plt.xlabel("x")
    #plt.ylabel("PDF")
    #plt.title("Beta Distributions")
    #plt.legend()
    #plt.grid(True)
    #plt.show()

    for d in range(0, 4):
        if d == 0:
            vout3_x2_05 = x1 * 0.5
        else:
            circ = C_FSM_SYNC(d)
            dv = np.array(sp.symbols('v0 v1 v2 v3', real=True, nonneg=True))
            T = circ.get_T(dv)
            pi = get_steady_state(T, vars=dv)
            ptm = circ.get_PTM(pi)
            vin = get_sym_ptv(np.array([[1, 0], [0, 1]]))
            for idx, v in enumerate(dv):
                ptm = ptm.subs(v, vin[idx])
            vout = sp.nsimplify(sp.Matrix(ptm.T @ vin))
            
            # Substitute x2=0.5 in vout[3] and plot the result with respect to x1
            vout3_x2_05 = vout[3].subs(x2, 0.5)
            vout3_vals = np.array([float(vout3_x2_05.subs(x1, val)) for val in x1_vals])
            plt.plot(x1_vals, np.sqrt((vout3_vals - ideal_vals) ** 2), label="FSM, depth={}".format(d))

        #Calculate the BMSE from the pdfs above
        funcs = [center_beta_pdf, uniform_pdf, left_bias_pdf, right_bias_pdf]
        names = ["Beta(3,3) Centered", "Beta(1,1) Uniform", "Beta(3,8) Left-biased", "Beta(8,3) Right-biased"]
        for func, name in zip(funcs, names):
            pdf_func = lambda x: float(func.subs(x1, x))
            mse_func = lambda x: (vout3_x2_05.subs(x1, x) - min(x, 0.5)) ** 2
            rbmse = np.round(np.sqrt(BMSE(pdf_func, mse_func)), 3)
            print("depth: {}, func: {}, RBMSE: {}".format(d, name, rbmse))

    xsq = np.array([float((x1 *0.5).subs(x1, val)) for val in x1_vals])
    plt.plot(x1_vals, np.sqrt((xsq - ideal_vals) ** 2), label="No synchronization")
    plt.xlabel("X")
    plt.ylabel("RMSE Error")
    plt.title("RMSE Error versus X")
    plt.grid(True)
    plt.legend()
    plt.show()