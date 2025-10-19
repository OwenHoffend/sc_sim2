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

def test_fsm_sync():
    """Simulation of FSM synchronizers with respect to depth d"""

    num_sccs = 100
    sccs = np.linspace(-1, 1, num_sccs)
    px = 0.25
    py = 0.25
    w = 12
    N = 2048
    for d in range(1, 4):
        print("Depth={}".format(d))
        input_sccs = []
        output_sccs = []
        sync = C_FSM_SYNC(d)
        for scc in sccs:
            cin_uncorr = np.eye(2)
            
            if scc < 0:
                cin_corr = -np.ones((2, 2)) + 2 * np.eye(2)
            else:
                cin_corr = np.ones((2, 2))
            sng = LFSR_SNG(w, C_WIRE(2, cin_uncorr))
            sng2 = LFSR_SNG(w, C_WIRE(2, cin_corr))
            #sng = RAND_SNG(w, C_WIRE(2, cin_uncorr))
            #sng2 = RAND_SNG(w, C_WIRE(2, cin_corr))
            bs_mat = sng.run(np.array([px, py]), N)
            bs_mat2 = sng2.run(np.array([px, py]), N)
            bs_mat_out = nonint_scc(bs_mat, bs_mat2, scc)
            input_scc = scc_mat(bs_mat_out)[0, 1]
            input_sccs.append(input_scc)
            bs_mat_sync = sync.run(bs_mat_out)
            output_scc = scc_mat(bs_mat_sync)[0, 1]
            output_sccs.append(output_scc)
        if d == 1:      
            plt.scatter(sccs, input_sccs, label="Input SCC")
        plt.scatter(sccs, output_sccs, label="Output SCC depth={}".format(d))
    plt.title("Input SCC vs Output SCC")
    plt.xlabel("Test SCC")
    plt.ylabel("Output SCC")
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
        ptm = circ.get_PTM_steady_state(pi)
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

def test_symbolic_fsm():
    for d in range(1, 4):
        circ = C_FSM_SYNC(d)
        dv = np.array(sp.symbols('v0 v1 v2 v3', real=True, nonneg=True))
        T = circ.get_T(dv)
        pi = get_steady_state(T, vars=dv)
        ptm = circ.get_PTM_steady_state(pi)
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