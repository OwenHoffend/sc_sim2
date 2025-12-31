import sympy as sp
from sympy.physics.quantum import TensorProduct
import numpy as np
from symb_analysis.seq_CAP import FSM_to_transition_matrix, extend_markov_chain_t1, get_steady_state, get_steady_state_nullspace, get_DV_symbols, get_dv_from_rho_single, lfsr_dv_model, get_extended_mealy_ptm, numeric_seq_CAP
from sim.PTV import get_Q, get_actual_DV_1cycle
import matplotlib.pyplot as plt
from sim.SCC import ascc_prob, ascc_from_bs, scc
from sim.SNG import LFSR_SNG, RAND_SNG
from sim.circs.circs import C_AND_N
from sim.visualization import plot_scc_heatmap
from sim.circs.tanh import C_TANH
from sim.circs.circs import C_WIRE
from sim.circs.SCMCs import C_FSM_SYNC
from sim.circs.misc_seq import C_DFF_MEALY
from sim.Util import sympy_vector_kron

def test_get_steady_state():
    x, y = sp.symbols('x y', real=True, nonneg=True)

    #Validated it on these three test cases from Armin's Exploiting Correlation paper:
    #T = sp.Matrix([[1-x, x, 0], [1-x, 0, x], [1-x, 0, x]])
    
    #T = sp.Matrix([
    #    [1-x, x, 0, 0],
    #    [1-x, 0, x, 0],
    #    [0, 1-x, 0, x],
    #    [0, 0, 1-x, x]
    #])

    #T = sp.Matrix([
    #    [1-x*y, x*y],
    #    [(1-x)*(1-y), x+y-x*y]
    #])

    #Transition matrix for a D-flipflop modeled with zero time steps of history
    #T = sp.Matrix([
    #    [1-x, x],
    #    [1-x, x]
    #])
    #print(get_steady_state(T))

    #Transition matrix for a D-flipflop modeled with one time step of history
    #dv = np.array(sp.symbols('v0 v1 v2 v3', real=True, nonneg=True))
    #T = sp.Matrix([
    #    [dv[0]/(dv[0]+dv[2]), dv[2]/(dv[0]+dv[2]), 0, 0],
    #    [0, 0, dv[1]/(dv[1]+dv[3]), dv[3]/(dv[1]+dv[3])],
    #    [dv[0]/(dv[0]+dv[2]), dv[2]/(dv[0]+dv[2]), 0, 0],
    #    [0, 0, dv[1]/(dv[1]+dv[3]), dv[3]/(dv[1]+dv[3])],
    #])
    #print(get_steady_state(T))

    #Transition matrix for a D=1 FSM synchronizer with one time step of history
    

    #print(get_steady_state(T))
    
def lfsr_autocorrelation_simulation_1d():
    #Run a couple LFSRs and measure autocorrelation properties
    w = 10
    sng = LFSR_SNG(w, C_WIRE(1, np.eye(1)))
    poly_inds = [0, 1, 2]
    px_values = np.linspace(0, 1, 1000)
    
    for poly_ind in poly_inds:
        xb_xbs = []
        xb_xs = []
        x_xbs = []
        x_xs = []
        for px in px_values:
            parr = [px]
            bs_mat = sng.run(parr, 2 ** w - 1, use_rand_init=False, poly_idx=poly_ind, add_zero_state=False)
            #bs_mat = sng.run(parr, 2 ** w - 1)
            xb_xb, xb_x, x_xb, x_x = get_actual_DV_1cycle(bs_mat)
            xb_xbs.append(xb_xb)
            xb_xs.append(xb_x)
            x_xbs.append(x_xb)
            x_xs.append(x_x)
        plt.plot(px_values, xb_xbs, label="xb_xbs")
        plt.plot(px_values, xb_xs, label="xb_xs")
        plt.plot(px_values, x_xbs, label="x_xbs")
        plt.plot(px_values, x_xs, label="x_xs")
        plt.legend()
        plt.show()

def lfsr_autocorrelation_simulation_2d():
    #Run a couple LFSRs and measure autocorrelation properties
    w = 5
    Cin = np.eye(2)
    circ = C_AND_N(2, Cin)
    sng = LFSR_SNG(w, circ)
    px_values = np.linspace(0, 1, 2 ** w)
    py_values = np.linspace(0, 1, 2 ** w)

    # Prepare matrices for each value of interest
    scc_matrix = np.zeros((2 ** w, 2 ** w))
    scc_xtm1_ytm1_matrix = np.zeros((2 ** w, 2 ** w))
    ascc_xtm1_y_matrix = np.zeros((2 ** w, 2 ** w))
    ascc_x_ytm1_matrix = np.zeros((2 ** w, 2 ** w))
    ascc_x_matrix = np.zeros((2 ** w, 2 ** w))
    ascc_y_matrix = np.zeros((2 ** w, 2 ** w))

    #Prepare matrices for saving the vin values for each pair of px and py
    vin_matrix = np.zeros((2 ** w, 2 ** w, 4))

    for i, px in enumerate(px_values):
        for j, py in enumerate(py_values):
            parr = [px, py]
            bs_mat = sng.run(parr, 2 ** (w*2) - 1, use_rand_init=False, poly_idx=0, add_zero_state=False)
            bsx = bs_mat[0, :]
            bsy = bs_mat[1, :]
            scc_value = scc(bsx, bsy)
            scc_xtm1_ytm1 = scc(np.roll(bsx, 1), np.roll(bsy, 1))
            ascc_xtm1_y = scc(np.roll(bsx, 1), bsy)
            #ascc_xtm1_y = scc(bsx[:-1], bsy[1:])
            ascc_x_ytm1 = scc(bsx, np.roll(bsy, 1))
            #ascc_x_ytm1 = scc(bsx[1:], bsy[:-1])
            ascc_x = ascc_from_bs(bsx)
            ascc_y = ascc_from_bs(bsy)

            scc_matrix[i, j] = scc_value
            scc_xtm1_ytm1_matrix[i, j] = scc_xtm1_ytm1
            ascc_xtm1_y_matrix[i, j] = ascc_xtm1_y
            ascc_x_ytm1_matrix[i, j] = ascc_x_ytm1
            ascc_x_matrix[i, j] = ascc_x
            ascc_y_matrix[i, j] = ascc_y

            #calculate vin values
            #The question I'm answering with this is: when using the simulated vin for a single cycle of delay,
            # as input to the analytical model,
            # does the analytical prediction for the tanh function match fully-simulated one?
            vin_matrix[i, j, 0] = np.mean(np.bitwise_and(np.roll(bsx, 1), np.roll(bsy, 1)))
            vin_matrix[i, j, 1] = np.mean(np.bitwise_and(bsx, np.roll(bsy, 1))) #not sure what the order of these should be
            vin_matrix[i, j, 2] = np.mean(np.bitwise_and(np.roll(bsx, 1), bsy))
            vin_matrix[i, j, 3] = np.mean(np.bitwise_and(bsx, bsy))

    #plot_scc_heatmap(scc_matrix, px_values, py_values, title="SCC(bsx, bsy)", xlabel="px", ylabel="py")
    #plot_scc_heatmap(scc_xtm1_ytm1_matrix, px_values, py_values, title="SCC(bs_x[t-1], bs_y[t-1])", xlabel="px", ylabel="py")
    #plot_scc_heatmap(ascc_xtm1_y_matrix, px_values, py_values, title="ASCC(bs_x[t-1], bs_y[t])", xlabel="px", ylabel="py")
    #plot_scc_heatmap(ascc_x_ytm1_matrix, px_values, py_values, title="ASCC(bs_x[t], bs_y[t-1])", xlabel="px", ylabel="py")
    #plot_scc_heatmap(ascc_x_matrix, px_values, py_values, title="ASCC(bs_x)", xlabel="px", ylabel="py")
    #plot_scc_heatmap(ascc_y_matrix, px_values, py_values, title="ASCC(bs_y)", xlabel="px", ylabel="py")

def test_seq_dv():
    #Get the DVs for -1, 0, and 1 auto-SCC
    x, xx = sp.symbols('x xx', real=True, nonneg=True)
    Q = get_Q(2, lsb='right')
    Qinv = sp.Matrix(np.linalg.inv(Q))
    vn1 = sp.nsimplify(Qinv @ sp.Matrix([1, x, x, sp.Max(2*x-1, 0)]))
    v0 = sp.nsimplify(Qinv @ sp.Matrix([1, x, x, x**2]))
    v1 = sp.nsimplify(Qinv @ sp.Matrix([1, x, x, x]))
    print(sp.latex(vn1))
    print(sp.latex(v0))
    print(sp.latex(v1))

def test_FSM_DFF():
    #Test of extended Markov chain on D-flipflop
    x, xb = sp.symbols("x xb")
    transitions = [(0, 1, x), (0, 0, xb), (1, 0, xb), (1, 1, x)]
    transitions = extend_markov_chain_t1(transitions, ["x"])

    T = FSM_to_transition_matrix(4, transitions, vars=["x"])
    print(get_steady_state_nullspace(T))

def test_FSM_SYNC():
    #Test of extended Markov chain on FSM synchronizer
    vars = ["x", "y"]
    circ = C_FSM_SYNC(1)
    transitions = circ.get_transition_list()

    #Here we substitute the DV variables with the actual probabilities
    #This is done to reduce the number of different symbols in the transition matrix
    #Otherwise the steady state solution will be very slow to compute

    #Bernoulli DV model
    #dv_x = get_dv_from_rho_single(0, symbol=sp.symbols("x"))
    #dv_y = get_dv_from_rho_single(0, symbol=sp.symbols("y"))

    #LFSR DV model
    dv_x = lfsr_dv_model(1, symbol=sp.symbols("x"))
    dv_y = lfsr_dv_model(1, symbol=sp.symbols("y"))

    #dv = sympy_vector_kron(dv_x, dv_y)
    #For the time being, hardcode the kronecker product for the joint DV with respect to the x and y DVs
    dv = sp.Matrix([
        dv_x[0] * dv_y[0],
        dv_x[0] * dv_y[1],
        dv_x[1] * dv_y[0],
        dv_x[1] * dv_y[1],
        dv_x[0] * dv_y[2], 
        dv_x[0] * dv_y[3], 
        dv_x[1] * dv_y[2],
        dv_x[1] * dv_y[3],
        dv_x[2] * dv_y[0],
        dv_x[2] * dv_y[1],
        dv_x[3] * dv_y[0],
        dv_x[3] * dv_y[1],
        dv_x[2] * dv_y[2],
        dv_x[2] * dv_y[3],
        dv_x[3] * dv_y[2],        
        dv_x[3] * dv_y[3],                
    ])

    extended_transitions, state_mapping = extend_markov_chain_t1(transitions, vars, dv=dv, return_state_mapping=True)
    #print(transitions)
    T = FSM_to_transition_matrix(max(state_mapping.keys()) + 1, extended_transitions, vars=vars, time_steps=1)

    pi = get_steady_state_nullspace(T)
    #pi = sp.symbols(" ".join([f"pi{x}" for x in range(len(state_mapping.keys()))]))

    mealy_TTs = circ.get_mealy_TTs()
    extended_ptm = get_extended_mealy_ptm(pi, transitions, extended_transitions, state_mapping, vars, mealy_TTs)

    x_vals = np.linspace(0, 1, 250)
    y = 0.5

    #Simulate the circuit
    w = 10
    #sng = RAND_SNG(w, C_WIRE(2, np.eye(2)))
    sng = LFSR_SNG(w, C_WIRE(2, np.eye(2)))
    xys = []
    for x in x_vals:
        bs_mat = sng.run([x, y], 2 ** w)
        bs_out = circ.run(bs_mat)
        xys.append(np.mean(np.bitwise_and(bs_out[0, :], bs_out[1, :])))

    #Analyze the circuit
    xys_analytic = []
    dv = dv.subs("y", y)
    extended_ptm = extended_ptm.subs("y", y)
    for x in x_vals:
        print(x)        
        vout = extended_ptm.T.subs("x", x) @ dv.subs("x", x)
        xys_analytic.append(sum(vout[-4:]))
    plt.plot(x_vals, xys, label="xys")
    plt.plot(x_vals, xys_analytic, label="xys_analytic")
    plt.legend()
    plt.show()

def test_FSM_SYNC_numeric():
    #Test of extended Markov chain on FSM synchronizer
    circ = C_FSM_SYNC(1)
    num_x_vals = 100
    x_vals = np.linspace(0.01, 0.98, num_x_vals)
    y = 0.5

    #Simulate the circuit
    w = 10
    #sng = RAND_SNG(w, C_WIRE(2, np.eye(2)))
    sng = LFSR_SNG(w, C_WIRE(2, np.eye(2)))
    xys = []
    for x in x_vals:
        bs_mat = sng.run([x, y], 2 ** w)
        bs_out = circ.run(bs_mat)
        xys.append(np.mean(np.bitwise_and(bs_out[0, :], bs_out[1, :])))

    #Bernoulli DV model
    dv_x = get_dv_from_rho_single(0, symbol=sp.symbols("x0"))
    dv_y = get_dv_from_rho_single(0, symbol=sp.symbols("x1"))

    #LFSR DV model
    #dv_x = lfsr_dv_model(1, symbol=sp.symbols("x0"))
    #dv_y = lfsr_dv_model(1, symbol=sp.symbols("x1"))

    #dv = sympy_vector_kron(dv_x, dv_y)
    #For the time being, hardcode the kronecker product for the joint DV with respect to the x and y DVs
    dv = sp.Matrix([
        dv_x[0] * dv_y[0],
        dv_x[0] * dv_y[1],
        dv_x[1] * dv_y[0],
        dv_x[1] * dv_y[1],
        dv_x[0] * dv_y[2], 
        dv_x[0] * dv_y[3], 
        dv_x[1] * dv_y[2],
        dv_x[1] * dv_y[3],
        dv_x[2] * dv_y[0],
        dv_x[2] * dv_y[1],
        dv_x[3] * dv_y[0],
        dv_x[3] * dv_y[1],
        dv_x[2] * dv_y[2],
        dv_x[2] * dv_y[3],
        dv_x[3] * dv_y[2],        
        dv_x[3] * dv_y[3],                
    ])

    parrs = [[x, y] for x in x_vals]
    vouts = numeric_seq_CAP(circ, dv, parrs, mode="ptv")
    xys_numeric = []
    for vout in vouts:
        xys_numeric.append(sum(vout[-4:]))

    plt.plot(x_vals, xys, label="xys")
    plt.plot(x_vals, xys_numeric, label="xys_numeric")
    plt.legend()
    plt.show()

def test_FSM_SYNC_input_output_SCC():
    #Test of extended Markov chain on FSM synchronizer
    circ = C_FSM_SYNC(1)
    num_x_vals = 100
    x_vals = np.linspace(0.01, 0.98, num_x_vals)
    y = 0.5

    #Simulate the circuit
    w = 10
    #sng = RAND_SNG(w, C_WIRE(2, np.eye(2)))
    sng = LFSR_SNG(w, C_WIRE(2, np.eye(2)))
    xys = []
    for x in x_vals:
        bs_mat = sng.run([x, y], 2 ** w)
        bs_out = circ.run(bs_mat)
        xys.append(np.mean(np.bitwise_and(bs_out[0, :], bs_out[1, :])))

    #Bernoulli DV model
    dv_x = get_dv_from_rho_single(0, symbol=sp.symbols("x0"))
    dv_y = get_dv_from_rho_single(0, symbol=sp.symbols("x1"))

    #LFSR DV model
    #dv_x = lfsr_dv_model(1, symbol=sp.symbols("x0"))
    #dv_y = lfsr_dv_model(1, symbol=sp.symbols("x1"))

    #dv = sympy_vector_kron(dv_x, dv_y)
    #For the time being, hardcode the kronecker product for the joint DV with respect to the x and y DVs
    dv = sp.Matrix([
        dv_x[0] * dv_y[0],
        dv_x[0] * dv_y[1],
        dv_x[1] * dv_y[0],
        dv_x[1] * dv_y[1],
        dv_x[0] * dv_y[2], 
        dv_x[0] * dv_y[3], 
        dv_x[1] * dv_y[2],
        dv_x[1] * dv_y[3],
        dv_x[2] * dv_y[0],
        dv_x[2] * dv_y[1],
        dv_x[3] * dv_y[0],
        dv_x[3] * dv_y[1],
        dv_x[2] * dv_y[2],
        dv_x[2] * dv_y[3],
        dv_x[3] * dv_y[2],        
        dv_x[3] * dv_y[3],                
    ])

    parrs = [[x, y] for x in x_vals]
    vouts = numeric_seq_CAP(circ, dv, parrs, mode="ptv")
    xys_numeric = []
    for vout in vouts:
        xys_numeric.append(sum(vout[-4:]))

    plt.plot(x_vals, xys, label="xys")
    plt.plot(x_vals, xys_numeric, label="xys_numeric")
    plt.legend()
    plt.show()

def test_FSM_TANH():
    #Test the DFF design from [Baker & Hayes, 2019]
    x, xb, m, r = sp.symbols("x xb m r")
    rhos = [0]
    #rhos = [-1, -0.5, 0, 0.5, 1]
    rho_out_curves = []  # Store all rho_out curves for plotting
    pout_sim_curves = []
    pout_analytic_curves = []
    for rho in rhos:
        print("rho =", rho)
        #dv = get_dv_from_rho_single(rho, use_new_symbol_for_max=True)
        dv = lfsr_dv_model(1)
        
        transitions = [(0, 0, xb), (0, 1, x), (1, 0, xb), (1, 2, x), (2, 1, xb), (2, 3, x), (3, 2, xb), (3, 3, x)]
        transitions = extend_markov_chain_t1(transitions, ["x"], dv=dv)
        #print(transitions)

        T = FSM_to_transition_matrix(8, transitions, vars=["x"])
        pi = get_steady_state_nullspace(T)

        vout = [sp.simplify(pi[0] + pi[1] + pi[2]), pi[3], pi[4], sp.simplify(pi[5] + pi[6] + pi[7])]
        pout = vout[3] + vout[2]
        pxt1xt2 = vout[3]
        if rho < 0:
            pout = sp.simplify(pout).subs(m, sp.Max(2*x-1, 0))
            pxt1xt2 = sp.simplify(pxt1xt2).subs(m, sp.Max(2*x-1, 0))
        #print(sp.latex(sp.simplify(pout)))
        #print(sp.latex(sp.simplify(ascc_prob(pout, pxt1xt2))))

        correct = 0.5 * (1 + sp.tanh(2*(2*x-1))) #stochastic tanh function

        x_vals = np.linspace(0, 1, 1000)
        pout_vals = [pout.subs(x, x_val) for x_val in x_vals]
        #plt.plot(x_vals, pout_vals, label="analytical, rho={}".format(rho))
        plt.plot(x_vals, pout_vals, label="analytical, LFSR DV model")

        # Compute rho_out and save for later plotting
        rho_out = []
        for i,x_val in enumerate(x_vals):
            pxt1t2_val = pxt1xt2.subs(x, x_val)
            if (pout_vals[i] == sp.nan or pxt1t2_val == sp.nan):
                rho_out.append(np.nan)
                continue
            result = ascc_prob(pout_vals[i], pxt1t2_val)
            if result == sp.zoo:
                rho_out.append(np.nan)
            else:
                rho_out.append(result)
        rho_out_curves.append((rho, list(rho_out)))

        #Simulate the circuit with real LFSRs
        circ = C_TANH(2)
        w = 10
        sng = LFSR_SNG(w, C_WIRE(1, np.eye(1)))
        #sng = RAND_SNG(w, C_WIRE(1, np.eye(1)))
        poly_inds = [0,]
        for poly_ind in poly_inds:
            pout_sim_curve = []
            for x_val in x_vals:
                print(x_val)
                parr = [x_val]
                bs_mat = sng.run(parr, 2 ** w - 1, use_rand_init=False, poly_idx=poly_ind, add_zero_state=False)
                #bs_mat = sng.run(parr, 2 ** w - 1)
                bs_out = circ.run(bs_mat)
                pout_sim_curve.append(np.mean(bs_out))
                
            pout_sim_curves.append(pout_sim_curve)

        #Apply autocorrelation model using LFSR dv model

    #Plot the pout curves
    plt.plot(x_vals, [correct.subs(x, x_val) for x_val in x_vals], label="correct")
    for poly_ind, pout_sim_curve in zip(poly_inds, pout_sim_curves):
        plt.plot(x_vals, pout_sim_curve, label="simulated, poly_ind={}".format(poly_ind))
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("z vs x")
    plt.legend()
    plt.show()

    # Now plot rho_out curves
    plt.figure()
    for rho, rho_out_vals in rho_out_curves:
        for rho_out_val in rho_out_vals:
            print(f"rho_out_val: {rho_out_val}, type: {type(rho_out_val)}")
        plt.plot(x_vals, rho_out_vals, label="rho={}".format(rho))
    plt.xlabel("x")
    plt.ylabel("rho_out")
    plt.title("rho_out vs x")
    plt.legend()
    plt.show()

def test_get_extended_mealy_ptm_DFF():
    #We will test with a DFF that has a Mealy output in its two states
    #x, xb = sp.symbols("x xb")
    #transitions = [(0, 1, x), (0, 0, xb), (1, 0, xb), (1, 1, x)]
    #mealy_TTs = sp.Matrix([[1, 0], [0, 1]]) #s0 = x', s1 = x
    circ = C_DFF_MEALY()
    transitions = circ.get_transition_list()
    mealy_TTs = circ.get_mealy_TTs()

    vars = ["x"]
    #dv = get_dv_from_rho_single(0)
    dv = lfsr_dv_model(1)
    extended_transitions, state_mapping = extend_markov_chain_t1(transitions, vars, dv=dv, return_state_mapping=True)
    T = FSM_to_transition_matrix(4, extended_transitions, vars=["x"])
    #pi = sp.symbols(" ".join([f"pi{x}" for x in range(len(state_mapping.keys()))]))
    pi = get_steady_state_nullspace(T)
    extended_ptm = get_extended_mealy_ptm(pi, transitions, extended_transitions, state_mapping, vars, mealy_TTs)

    #Now test the extended PTM model
    x_vals = np.linspace(0, 1, 1000)

    #Simulate the circuit
    w = 10
    #sng = RAND_SNG(w, C_WIRE(1, np.eye(1)))
    sng = LFSR_SNG(w, C_WIRE(1, np.eye(1)))
    xb_xbs = []
    xb_xs = []
    x_xbs = []
    x_xs = []
    for x in x_vals:
        bs_mat = sng.run(x, 2 ** w)
        bs_out = circ.run(bs_mat)
        xb_xb, xb_x, x_xb, x_x = get_actual_DV_1cycle(bs_out)
        xb_xbs.append(xb_xb)
        xb_xs.append(xb_x)
        x_xbs.append(x_xb)
        x_xs.append(x_x)
    plt.plot(x_vals, xb_xbs, label="xb_xbs")
    plt.plot(x_vals, xb_xs, label="xb_xs")
    plt.plot(x_vals, x_xbs, label="x_xbs")
    plt.plot(x_vals, x_xs, label="x_xs")
    plt.legend()
    plt.show()   

    #Analyze the circuit
    xb_xbs = []
    xb_xs = []
    x_xbs = []
    x_xs = []
    for x_val in x_vals:
        vout = extended_ptm.subs("x", x_val).T @ dv.subs("x", x_val)
        xb_xbs.append(vout[0])
        xb_xs.append(vout[1])
        x_xbs.append(vout[2])
        x_xs.append(vout[3])

    plt.plot(x_vals, xb_xbs, label="xb_xbs")
    plt.plot(x_vals, xb_xs, label="xb_xs")
    plt.plot(x_vals, x_xbs, label="x_xbs")
    plt.plot(x_vals, x_xs, label="x_xs")
    plt.legend()
    plt.show()  

def test_get_extended_mealy_ptm_SYNC():
    circ = C_FSM_SYNC(1)
    transitions = circ.get_transition_list()
    vars = ["x", "y"]
    transitions, state_mapping = extend_markov_chain_t1(transitions, vars, return_state_mapping=True)
    #get_extended_mealy_ptm(transitions, state_mapping, vars)
