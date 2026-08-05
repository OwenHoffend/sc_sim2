import numpy as np
import sympy as sp
from sim.circs.muller_c import C_MULLER_C, C_MULLER_C_cascade
from symb_analysis.seq_CAP import *
from symb_analysis.CAP import get_sym_ptv
import matplotlib.pyplot as plt
from sim.SNG import LFSR_SNG, RAND_SNG
from sim.SCC import ascc_from_bs, ascc_prob, scc

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

    rho_x = 0
    rho_y = 0

    #Bernoulli DV model
    dv_x = get_dv_from_rho_single(rho_x, symbol=vars[0])
    dv_y = get_dv_from_rho_single(rho_y, symbol=vars[1])

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

    transitions = circ.get_transition_list()
    extended_transitions, state_mapping = extend_markov_chain_t1(transitions, vars, dv=dv, return_state_mapping=True)
    T = FSM_to_transition_matrix(max(state_mapping.keys()) + 1, extended_transitions, vars=vars, time_steps=1)
    pi = get_steady_state_nullspace(T)

    extended_ptm = sp.Matrix([[pi[i] for i in range(len(pi))] for _ in range(16)])

    #Simulate the circuit
    w = 13
    sng = RAND_SNG(w, np.eye(2))
    #sng = LFSR_SNG(w, np.eye(2))

    # Sweep both x and y, show 3D surface plots
    x_vals = np.linspace(0.001, 0.999, 15)
    y_vals = np.linspace(0.001, 0.999, 15)
    X, Y = np.meshgrid(x_vals, y_vals)

    #input stats
    input_X_asccs_sim = np.zeros_like(X)
    input_Y_asccs_sim = np.zeros_like(X)
    input_scc_sim = np.zeros_like(X)
    input_asccs_analytic = np.zeros_like(X)

    #output stats
    zs = np.zeros_like(X)
    asccs_sim = np.zeros_like(X)
    correct_values = np.zeros_like(X, dtype='float')
    zs_analytic = np.zeros_like(X, dtype='float')
    asccs_analytic = np.zeros_like(X, dtype='float')

    for i in range(X.shape[0]):
        print(f"{i} of {X.shape[0]}")
        for j in range(X.shape[1]):
            
            # Simulated results
            x = X[i, j]
            y = Y[i, j]
            bs_mat = sng.run([x, y], 2 ** w, rhos=[rho_x, rho_y])
            
            input_X_asccs_sim[i, j] = ascc_from_bs(bs_mat[0, :])
            input_Y_asccs_sim[i, j] = ascc_from_bs(bs_mat[1, :])
            input_scc_sim[i, j] = scc(bs_mat[0, :], bs_mat[1, :])

            #Simulated function values
            bs_out = circ.run(bs_mat)
            zs[i, j] = np.mean(bs_out)
            asccs_sim[i, j] = ascc_from_bs(bs_out)

            # Correct function values
            correct_values[i, j] = circ.correct(bs_mat[0, :], bs_mat[1, :])

            # Analytic function values
            dv_xy = dv.subs("x0", x).subs("x1", y)
            input_asccs_analytic[i, j] = ascc_prob(sum(dv_xy[-8:]), dv_xy[10] + dv_xy[11] + dv_xy[14] + dv_xy[15])
            ptm_xy = extended_ptm.subs("x0", x).subs("x1", y)
            vout = ptm_xy.T @ dv_xy
            z_out = vout[2] + vout[3]
            zs_analytic[i, j] = z_out
            asccs_analytic[i, j] = ascc_prob(z_out, vout[3])

    # Compute RMSE between simulated z_out and correct values
    rmse_output_sim = np.sqrt(np.mean((zs - correct_values) ** 2))

    # Compute RMSE between analytic z_out and correct values
    rmse = np.sqrt(np.mean((zs_analytic - correct_values) ** 2))

    # Compute RMSE between analytic and simulated output mean values
    rmse_output = np.sqrt(np.mean((zs_analytic - zs) ** 2))

    # Compute RMSE between analytic and simulated autocorrelation
    # Flatten the arrays to 1D
    analytic_flat = asccs_analytic.flatten()
    sim_flat = asccs_sim.flatten()
    # Mask positions where either value is nan or -1
    mask = (
        ~np.isnan(analytic_flat) & ~np.isnan(sim_flat) &
        (analytic_flat != -1) & (sim_flat != -1)
    )
    diffs = analytic_flat[mask] - sim_flat[mask]
    rmse_ascc = np.sqrt(np.mean(diffs ** 2))

    print(f"RMSE between simulated z_out and correct values: {rmse_output_sim}")
    print(f"RMSE between analytic z_out and correct values: {rmse}")
    print(f"RMSE between analytic z_out and simulated: {rmse_output}")
    print(f"RMSE between analytic asccs and simulated: {rmse_ascc}")

    # Plot surface for input ascc (simulated and analytic)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, input_X_asccs_sim, cmap='Reds', label="simulated X")
    ax.plot_surface(X, Y, input_Y_asccs_sim, cmap='Greens', label="simulated Y")
    ax.plot_surface(X, Y, input_scc_sim, cmap='Blues', label="simulated")
    ax.set_title("Input ASCCs")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("ASCC(input)")
    plt.legend(["Simulated X ASCC (red)", "Simulated Y ASCC (green)", "Simulated SCC (blue)"])
    plt.show()

    # Plot surface for zs_analytic only
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, zs_analytic, cmap='Reds')
    ax.plot_surface(X, Y, correct_values, cmap='Blues', label="correct_values")
    ax.plot_surface(X, Y, zs, cmap='Greens', label="simulated")
    ax.set_title("z output: analytic")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.legend(["Analytic (red)", "Correct (blue)", "Simulated (green)"])
    plt.show()

    # Plot surface for asccs_analytic only
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, asccs_analytic, cmap='Greens', label="analytic")
    ax.plot_surface(X, Y, asccs_sim, cmap='Oranges', label="simulated")
    ax.set_title("z autocorr: analytic")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("ascc(z)")
    plt.legend(["Analytic (green)", "Simulated (orange)"])
    plt.show()

def muller_c_cascade_test(k: int):
    

    #Simulate the circuit
    w = 12
    num_samples = 50
    sng = RAND_SNG(w, np.eye(k))

    sample_inputs = np.random.uniform(0, 1, (num_samples, k))

    correct_values = np.zeros((num_samples, k-1))
    for i in range(num_samples):
        print(f"Sample {i} of {num_samples}")
        for j in range(k-1):
            circ = C_MULLER_C_cascade(i)
            correct_values[i, j] = circ.correct(sample_inputs[i, :j])

    circ = C_MULLER_C_cascade(k)
    simulated_values = np.zeros(num_samples)
    simulated_asccs = np.zeros((num_samples, k-1))
    simulated_errs = np.zeros((num_samples, k-1))
    simulated_errs_rhos = np.zeros((num_samples, k-1))
    rhos = [0, 0.47717359, 0.69653191, 0.77210278, 0.83320094, 0.87416808, 0.88713954, 0.90578726, 0.92009654, 0.92844163]
    for i in range(num_samples):
        print(f"Sample {i} of {num_samples}")
        parr = sample_inputs[i, :]
        bs_mat = sng.run(parr, 2 ** w)
        bs_mat_rhos = sng.run(parr, 2 ** w, rhos=rhos)
        z, asccs, errs = circ.run(bs_mat, regen_between_layers=False, measure_autocorrelation_between_layers=True, measure_err_between_layers=True, w=w, correct_values=correct_values[i, :])
        z_rhos, asccs_rhos, errs_rhos = circ.run(bs_mat_rhos, regen_between_layers=False, measure_autocorrelation_between_layers=True, measure_err_between_layers=True, w=w, correct_values=correct_values[i, :])
        simulated_values[i] = np.mean(z)
        simulated_asccs[i] = asccs
        simulated_errs[i] = errs
        simulated_errs_rhos[i] = errs_rhos
    # Compute RMSE between simulated z_out and correct values
    #rmse = np.sqrt(np.mean((simulated_values - correct_values) ** 2))
    #print(f"RMSE between simulated z_out and correct values: {rmse}")
    layers = np.arange(1, k)  # Layers indexed from 1 to k-1 (autocorrelation between each)

    # Plot average error (simulated_errs) between each layer as a function of the layer
    avg_errs = np.mean(simulated_errs, axis=0)
    avg_errs_rhos = np.mean(simulated_errs_rhos, axis=0)
    print(f"Average errors (cross entropy) between layers: {avg_errs}")
    print(f"Average errors (cross entropy) between layers with rhos: {avg_errs_rhos}")
    plt.figure()
    plt.plot(layers, avg_errs, marker='o', color='red')
    plt.plot(layers, avg_errs_rhos, marker='o', color='blue')
    plt.xlabel('Layer')
    plt.ylabel('Average Error (Cross Entropy)')
    plt.title('Average Cross Entropy Error Between Layers in Muller C Cascade')
    plt.grid()
    plt.show()
    
    # Plot average autocorrelation (simulated_asccs) between each layer as a function of the layer
    avg_asccs = np.mean(simulated_asccs, axis=0)
    print(f"Average autocorrelations between layers: {avg_asccs}")
    
    plt.figure()
    plt.plot(layers, avg_asccs, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Average Autocorrelation (ASCC)')
    plt.title('Average Autocorrelation Between Layers in Muller C Cascade')
    plt.grid()
    plt.show()