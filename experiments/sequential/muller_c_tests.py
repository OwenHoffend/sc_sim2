import numpy as np
import sympy as sp
from sim.circs.muller_c import C_MULLER_C
from symb_analysis.seq_CAP import *
from symb_analysis.CAP import get_sym_ptv
import matplotlib.pyplot as plt
from sim.SNG import LFSR_SNG, RAND_SNG
from sim.SCC import ascc_from_bs, ascc_prob

def single_muller_c_correlation_analysis():
    circ = C_MULLER_C()
    T = FSM_to_transition_matrix(circ.ns, circ.get_transition_list())
    pi = get_steady_state(T)
    ptm = circ.get_PTM(pi)

    #Output probabilities assuming input is uncorrelated
    #C = np.eye(2)
    #C = np.ones((2,2))

    C = np.ones((2,2))
    dv = get_sym_ptv(C)
    dv_symbs = get_DV_symbols(circ.get_vars(), 0)
    for idx, symb in enumerate(dv_symbs):
        ptm = ptm.subs(symb, dv[idx])

    dv_out_1 = sp.simplify(sp.Matrix(ptm.T @ dv))[0]


    C = np.eye(2)
    dv = get_sym_ptv(C)
    ptm = circ.get_PTM(pi)
    dv_symbs = get_DV_symbols(circ.get_vars(), 0)
    for idx, symb in enumerate(dv_symbs):
        ptm = ptm.subs(symb, dv[idx])

    dv_out_0 = sp.simplify(sp.Matrix(ptm.T @ dv))

    C = 2 * np.eye(2) - np.ones((2,2))
    dv = get_sym_ptv(C)
    ptm = circ.get_PTM(pi)
    dv_symbs = get_DV_symbols(circ.get_vars(), 0)
    for idx, symb in enumerate(dv_symbs):
        ptm = ptm.subs(symb, dv[idx])

    dv_out_n1 = sp.simplify(sp.Matrix(ptm.T @ dv))

    import matplotlib.pyplot as plt

    # Define the sympy variables
    x1, x2 = sp.symbols('x1 x2', real=True, nonneg=True)
    x2_vals = np.linspace(0, 1, 200)

    # Evaluate dv_out_1, dv_out_0, dv_out_n1 for x1=0.5, for each x2 in x2_vals
    dv_out_1_vals = [float(dv_out_1.subs({x1: 0.25, x2: val})) for val in x2_vals]
    dv_out_0_vals = [float(dv_out_0[0].subs({x1: 0.25, x2: val})) for val in x2_vals]
    dv_out_n1_vals = [float(dv_out_n1[0].subs({x1: 0.25, x2: val})) for val in x2_vals]

    plt.plot(x2_vals, dv_out_1_vals, label='C=1')
    plt.plot(x2_vals, dv_out_0_vals, label='C=0')
    plt.plot(x2_vals, dv_out_n1_vals, label='C=-1')
    plt.xlabel('x2')
    plt.ylabel('Output Probability')
    plt.title('Muller C Output Probabilities vs x2 (x1=0.25)')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Output probabilities assuming input is correlated

def muller_c_autocorrelation():
    circ = C_MULLER_C()
    vars = circ.get_vars()

    #Bernoulli DV model
    dv_x = get_dv_from_rho_single(0.99, symbol=vars[0])
    dv_y = get_dv_from_rho_single(0.99, symbol=vars[1])

    #LFSR DV model
    #dv_x = lfsr_dv_model(1, symbol=vars[0])
    #dv_y = lfsr_dv_model(1, symbol=vars[1])

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

    #Generic DV using v symbols
    #dv = sp.Matrix([sp.symbols(f"v{i}") for i in range(16)])

    #Simulate the circuit
    w = 13
    sng = RAND_SNG(w, np.eye(2))
    #sng = LFSR_SNG(w, np.eye(2))

    # Sweep both x and y, show 3D surface plots
    x_vals = np.linspace(0.001, 0.999, 15)
    y_vals = np.linspace(0.001, 0.999, 15)
    X, Y = np.meshgrid(x_vals, y_vals)

    #Plot autocorrelation properties of the input bitstreams first
    #y = 0.25
    #asccs_sim = []
    #for x in x_vals:
    #    bs_mat = sng.run([x, y], 2 ** w)
    #    asccs_sim.append(ascc_from_bs(bs_mat[0, :]))
    #plt.plot(x_vals, asccs_sim, label="asccs_sim")
#
    #asccs_analytic = []
    #for x in x_vals:
    #    vin = dv.subs("x0", x).subs("x1", y)
    #    asccs_analytic.append(ascc_prob(sum(vin[-8:]), vin[10] + vin[11] + vin[14] + vin[15]))
    #plt.plot(x_vals, asccs_analytic, label="asccs_analytic")
    #plt.legend()
    #plt.show()

    transitions = circ.get_transition_list()
    extended_transitions, state_mapping = extend_markov_chain_t1(transitions, vars, dv=dv, return_state_mapping=True)
    T = FSM_to_transition_matrix(max(state_mapping.keys()) + 1, extended_transitions, vars=vars, time_steps=1)
    pi = get_steady_state_nullspace(T)

    extended_ptm = sp.Matrix([[pi[i] for i in range(len(pi))] for _ in range(16)])

    # Simulated results
    zs = np.zeros_like(X)
    asccs_sim = np.zeros_like(X)
    for i in range(X.shape[0]):
        print(i)
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]
            bs_mat = sng.run([x, y], 2 ** w)
            bs_out = circ.run(bs_mat)
            zs[i, j] = np.mean(bs_out)
            asccs_sim[i, j] = ascc_from_bs(bs_out)

    #Correct function
    def correct_function(x, y):
        return x * y / (x * y + (1 - x) * (1 - y))

    # Correct function values
    correct_values = np.zeros_like(X, dtype='float')
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            correct_values[i, j] = correct_function(X[i, j], Y[i, j])

    # Analytic results
    zs_analytic = np.zeros_like(X, dtype='float')
    asccs_analytic = np.zeros_like(X, dtype='float')
    for i in range(X.shape[0]):
        print(i)
        for j in range(X.shape[1]):
            x = X[i, j]
            y = Y[i, j]
            # Substitute both variables
            dv_xy = dv.subs("x0", x).subs("x1", y)
            ptm_xy = extended_ptm.subs("x0", x).subs("x1", y)
            vout = ptm_xy.T @ dv_xy
            z_out = vout[2] + vout[3]
            zs_analytic[i, j] = z_out
            asccs_analytic[i, j] = ascc_prob(z_out, vout[3])

    # Compute RMSE between analytic z_out and correct values
    rmse = np.sqrt(np.mean((zs_analytic - correct_values) ** 2))
    print(f"RMSE between analytic z_out and correct values: {rmse}")

    # Plot surface for zs_analytic only
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, zs_analytic, cmap='Reds')
    ax.plot_surface(X, Y, correct_values, cmap='Blues', label="correct_values")
    ax.set_title("z output: analytic")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.legend(["Analytic (red)", "Correct (blue)"])
    plt.show()

    # Plot surface for asccs_analytic only
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, asccs_analytic, cmap='Greens')
    ax.set_title("z autocorr: analytic")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("ascc(z)")
    plt.legend(["Analytic (green)"])
    plt.show()

def pair_muller_c_scc():
    #Get the output correlation of a pair of Muller-C elements assuming 
    pass