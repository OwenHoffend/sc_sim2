import numpy as np
import scipy
import time
from statsmodels.distributions.copula.api import GaussianCopula
from statsmodels.distributions.copula.api import CopulaDistribution
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sim.PTV import *
from sim.PTM import get_func_mat
from symb_analysis.experiments.subcirc_ptm import and_or_example
from sim.SCC import SCC_to_Pearson, Pearson_to_SCC, scc, scc_prob
from scipy import stats, optimize
from sim.visualization import plot_scc_heatmap
from sim.circs.circs import C_Sobel, C_SobelMuxes, C_RCED, C_MAC_N
from sim.SNG import LFSR_SNG, NONINT_RAND_SNG, GAUSSIAN_COPULA_SNG
from sim.datasets import dataset_uniform, dataset_sweep_1d, dataset_stack, dataset_cameraman, dataset_center_beta, dataset_beta
from sim.sim import sim_circ, sim_circ_PTM, gen_correct
from synth.experiments.example_circuits_for_proposal import XOR_with_AND
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal

def gauss_copula_vs_linear_comb():
	lambda_ = 0.9
	#acoeffs = np.array([ #this one works correctly
	#	[1, 0, 0],
	#	[lambda_, 1-lambda_, 0],
	#	[0, 0, 1]
	#])

	#RNS choices with zero weight
	"""
	[0, 0, 0]
	[0, 0, 1]
	[0, 1, 0]
	[0, 1, 1]
	"""

	acoeffs = np.array([ #this one does not, the resulting DV does not even sum to 1
		[1, 0, 0],
		[lambda_, 0, 1-lambda_],
		[0, 1, 0]
	])

	#RNS choices with zero weight
	"""
	[0, 0, 0]
	[0, 0, 2]
	[0, 1, 0]
	[0, 1, 1]
	[0, 1, 2]
	"""

	#Issue is we need to force the matrix to be lower-left triangular

	signs = np.array([1, 1, 1])
	pxs = np.array([0.5, 0.5, 0.5])

	#First method: convex combination of Frechet-Hoeffding copulas
	vin = get_DV_from_acoeffs_and_signs(acoeffs, signs, pxs)
	P, Cin = get_C_from_v(vin, lsb='right', return_P=True)
	print(P)
	print(np.round(Cin, 4))
	print(vin)

	#Second method: gaussian copulas
	#C = np.array([
	#	[1, lambda_, 0],
	#	[lambda_, 1, 0],
	#	[0, 0, 1]
	#])
	#vin2 = get_DV_via_gaussian_copula(C, pxs)
	#print(vin2)

def gauss_copula_test_3d():
	pxs = [0.75, 0.5, 0.25]
	C = np.array([
		[1, 0.3, 0], 
		[0.3, 1, -0.7], 
		[0, -0.7, 1]
	])
	v = get_DV_via_gaussian_copula(C, pxs)
	print(np.round(v, 5))

def gauss_copula_test_MAC():
	num_tests = 1000
	lambdas = np.linspace(-1, 1, 30)
	c_avgs = []
	for lambda_ in lambdas:
		print(lambda_)
		C = np.array([
			[1, 0, lambda_, 0],
			[0, 1, 0, lambda_],
			[lambda_, 0, 1, 0],
			[0, lambda_, 0, 1]
		])
		c_avg = 0
		for _ in range(num_tests):
			w = np.random.uniform(size=(8,))
			x = np.random.uniform(size=(8,))
			z1 = 0.25 * np.sum(w[0:4] * x[0:4])
			z2 = 0.25 * np.sum(w[4:8] * x[4:8])
			
			pxy = 0
			for i in range(4):
				px = np.array([x[i], w[i], x[i+4], w[i+4]])
				pxy += get_gaussian_copula(C, px).cdf(px)
			pxy *= 0.25

			scc = scc_prob(z1, z2, pxy)
			c_avg += scc
		c_avg /= num_tests
		c_avgs.append(c_avg)
	# Plot c_avg vs lambda_
	plt.figure()
	plt.plot(lambdas, c_avgs)
	plt.xlabel("Input SCC")
	plt.ylabel("Output SCC")
	plt.title("Output SCC vs Input SCC")
	plt.grid(True)
	plt.show()

def MAC_ReLU_copula():
	num_samples = 500
	nX = 128
	ww = 6
	circ = C_MAC_N(nX, relu=True)

	#+1 here in the correlation matrix is for the 0.5 input to the relu gate
	Cin = scipy.linalg.block_diag(np.ones((nX + 1, nX + 1)), np.ones((nX, nX)))
	sng = LFSR_SNG(ww, Cin, nc=circ.nc)
	#ds = dataset_uniform(num_samples, 2 * nX + 1)
	ds = dataset_beta(num_samples, 2 * nX + 1, 3.5, 1)

	#0.5-valued input for ReLU
	ds.ds[:, nX] = 0.5 * np.ones((num_samples,))

	#N_values = 2 ** np.arange(6, 14, dtype=np.int64)
	N_values = [512,]

	# --- Gaussian copula model (independent of N) ---
	print("Running Gaussian copula model")
	time_start = time.time()
	copula_outputs = []
	sccs = []
	for i, xs in enumerate(ds):
		if i % 50 == 0 or i == num_samples - 1:
			print(f"{i}/{num_samples}")

		r = 0.5
		z = 0
		zr = 0
		for j in range(nX):
			#bipolar conversion?
			x = xs[j]
			w = xs[nX + j + 1] #+1 for relu, parameterize if necessary

			#activation
			wx = x * w 
			z += 1 - x - w + 2 * wx

			#relu
			xr = np.minimum(r, x)
			wr = r * w
			wxr = w * xr
			zr += r - wr - xr + 2 * wxr
		z /= nX
		zr /= nX
		a = z + r - zr
		scc = scc_prob(r, z, zr)
		sccs.append(scc)
		copula_outputs.append(a)
		#sccs.append(0)
	copula_outputs = np.array(copula_outputs)
	time_end = time.time()
	copula_runtime = time_end - time_start
	print("Gaussian copula model time: ", copula_runtime)
	print("SCC: ", np.mean(sccs))

	# --- Bitstream simulation sweep over N ---
	rmses_sim_copula = []
	sim_runtimes = []
	for N in N_values:
		print(f"Bitstream simulation N={N}")
		t0 = time.time()
		result = sim_circ(sng, circ, ds, Nset=int(N), loop_print=False)
		t1 = time.time()
		sim_runtimes.append(t1 - t0)
		rmse_sim_copula = np.sqrt(np.mean((result.out - copula_outputs) ** 2))
		rmses_sim_copula.append(rmse_sim_copula)
		print(f"  RMSE sim vs copula: {rmse_sim_copula:.6g}, runtime: {t1 - t0:.3f} s")

		rmse_correct_copula = np.sqrt(np.mean((result.correct - copula_outputs) ** 2))
		rmse_correct_sim = np.sqrt(np.mean((result.correct - result.out) ** 2))
		print("RMSE, correct vs copula (reference): ", rmse_correct_copula)
		print("RMSE, correct vs sim: ", rmse_correct_sim)

	
	fig, ax_rmse = plt.subplots()
	color_rmse = "C0"
	color_time = "C1"
	ax_rmse.set_xlabel("N (bitstream length)")
	ax_rmse.set_ylabel("RMSE (sim vs copula)")
	line_rmse, = ax_rmse.plot(N_values, rmses_sim_copula, "o-", color=color_rmse, label="RMSE sim vs copula")
	ax_rmse.tick_params(axis="y")
	ax_rmse.set_xscale("log", base=2)
	ax_rmse.grid(True, which="both", alpha=0.3)

	ax_time = ax_rmse.twinx()
	ax_time.set_ylabel("Bitstream simulation runtime (s)")
	line_time, = ax_time.plot(N_values, sim_runtimes, "s--", color=color_time, label="Sim runtime")
	color_copula = "C2"
	line_copula_rt, = ax_time.plot(
		N_values,
		np.full_like(N_values, copula_runtime, dtype=float),
		linestyle=":",
		marker="D",
		markersize=5,
		color=color_copula,
		lw=2,
		label=f"Copula model runtime ({copula_runtime:.2g} s)",
	)
	ax_time.tick_params(axis="y")

	ax_rmse.legend(handles=[line_rmse, line_time, line_copula_rt], loc="upper center")
	ax_rmse.set_title("MAC ReLU: copula vs stochastic simulation")
	fig.tight_layout()
	plt.show()

def test_mv_copula():
	copula = GaussianCopula(1, allow_singular=True)
	xs, ys = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))

	cdf = copula.cdf(np.array([xs.flatten(), ys.flatten()]).T)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	surf = ax.plot_surface(xs, ys, cdf.reshape(100, 100), cmap=cm.coolwarm,
						linewidth=0, antialiased=False)
	
	plt.show()

def cascaded_copula():
	rho = 0
	copula_in = GaussianCopula(rho, allow_singular=True)
	N = 100
	xs, ys = np.meshgrid(np.linspace(0.01, 0.99, N), np.linspace(0.01, 0.99, N))
	xs_flat, ys_flat = xs.flatten(), ys.flatten()

	cxys = copula_in.cdf(np.vstack([xs_flat, ys_flat]).T)
	pxs = copula_in.cdf(np.vstack([xs_flat, np.ones((N ** 2,))]).T)
	pys = copula_in.cdf(np.vstack([np.ones((N ** 2,)), ys_flat]).T)
	pins_xs = np.vstack([np.ones((N ** 2,)), pxs, pys, cxys])

	#Here, I'm just manually checking that the input p vector has the correct correlation
	#It should equal "rho" above, at least in the integer cases -1, 0, and 1.
	Q = get_Q(2, lsb='right')
	Q_inv = np.linalg.inv(Q)
	#for i in range(N ** 2):
	#	p = pins_xs[:, i]
	#	v = Q_inv @ p
	#	P, C = get_C_from_v(v, return_P=True)
	#	if not np.all(np.isnan(C)) and np.all(P > 0):
	#		#breakpoint here to check
	#		print(P)
	#		print(C)
	#		pass

	#Now calculate the output for a test circuit and do the same check
	Mf = get_func_mat(and_or_example, 2, 2)
	C_outs = []
	for i in range(N ** 2):
		p = pins_xs[:, i]
		v = Q_inv @ p
		vout = Mf.T @ v
		P, C = get_C_from_v(vout, return_P=True)
		C_outs.append(C[0, 1])
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	mean_C_out = np.mean(np.nan_to_num(C_outs))
	print(mean_C_out)

	C_outs_array = np.array(C_outs).reshape(N, N)
	fig, ax = plt.subplots()
	heatmap = ax.imshow(C_outs_array, extent=[0.01, 0.99, 0.01, 0.99], origin='lower', cmap=cm.coolwarm, aspect='auto', vmin=-1, vmax=1)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('Output Correlation Heatmap')
	plt.colorbar(heatmap, ax=ax, label='Correlation')
	plt.show()

def and_or_example_copula():
	Q = get_Q(2, lsb='left')
	Q_inv = np.linalg.inv(Q)
	target_corr = -0.9
	Mf = get_func_mat(and_or_example, 2, 2)
	z2_outs = []
	scc_outs = []
	for px in np.linspace(0.001, 0.999, 100):
		for py in np.linspace(0.001, 0.999, 100):
			rho_latent = optimize.brentq(
				lambda r: SCC_from_rho_bv_gaussian(r, px, py) - target_corr,
				-0.999, 0.999
			)
			copula = GaussianCopula(rho_latent, allow_singular=True)
			pxy = copula.cdf(np.array([px, py]))
			p = np.array([1, px, py, pxy])
			v = Q_inv @ p
			pout = Q @ Mf.T @ v
			z2_outs.append(pout[2])
			scc_outs.append(scc_prob(pout[1], pout[2], pout[3]))

	# Plot SCC as a 2D heatmap instead of a 3D surface
	scc_outs_array = np.array(scc_outs).reshape(100, 100)
	plot_scc_heatmap(scc_outs_array, np.linspace(0.001, 0.999, 100), np.linspace(0.001, 0.999, 100), title="SCC(bsx, bsy)", xlabel="px", ylabel="py")

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	xs, ys = np.meshgrid(np.linspace(0.01, 0.99, 100), np.linspace(0.01, 0.99, 100))
	surf = ax.plot_surface(xs, ys, np.array(z2_outs).reshape(100, 100), cmap=cm.coolwarm,
						linewidth=0, antialiased=False)
	plt.show()

def gauss_copula_test_2d():
	c = 0
	px = 0.75
	py = 0.25

	#target_corr = SCC_to_Pearson(c, px, py)
	target_corr = c



	import matplotlib.pyplot as plt
	import numpy as np

	# Plot SCC_from_rho_bv_gaussian as a function of rho
	rho_values = np.linspace(-0.999, 0.999, 200)
	corr_values = [SCC_from_rho_bv_gaussian(rho) for rho in rho_values]

	plt.figure()
	plt.plot(rho_values, corr_values, label='SCC vs rho')
	plt.xlabel('rho')
	plt.ylabel('SCC')
	plt.title('SCC vs rho for px={}, py={}'.format(px, py))
	plt.grid(True)

	# Add a horizontal line at y=0.5
	plt.axhline(y=0.5, color='r', linestyle='--', label='SCC = 0.5')

	# Find intersection point (rho, 0.5)
	corr_values_np = np.array(corr_values)
	# Indices where the curve crosses 0.5
	signs = corr_values_np - 0.5
	cross_idx = np.where(np.diff(np.sign(signs)))[0]
	if cross_idx.size > 0:
		# Linear interpolation for better accuracy
		i = cross_idx[0]
		rho1, rho2 = rho_values[i], rho_values[i+1]
		scc1, scc2 = corr_values_np[i], corr_values_np[i+1]
		# interpolate for rho where scc == 0.5
		rho_inter = rho1 + (0.5 - scc1) * (rho2 - rho1) / (scc2 - scc1)
		plt.plot(rho_inter, 0.5, 'ko', label='Intersection (rho={:.3f}, SCC=0.5)'.format(rho_inter))

	plt.legend()
	plt.show()

	rho_latent = optimize.brentq(
		lambda r: SCC_from_rho_bv_gaussian(r) - target_corr,
		-0.999, 0.999
	)

	copula = GaussianCopula(rho_latent, allow_singular=True)
	_ = copula.plot_pdf()
	plt.show()

	marginals = [stats.uniform(0, 1), stats.uniform(0, 1)]
	joint_dist = CopulaDistribution(copula=copula, marginals=marginals)
	pxy = copula.cdf(np.array([px, py]))
	rho_out = (pxy - px * py) / (np.sqrt(px * (1-px)) * np.sqrt(py * (1-py)))
	print("Calculated RHO: ", rho_out)
	print(Pearson_to_SCC(rho_out, px, py))

	samps = joint_dist.rvs(100000)
	samps_thresh = np.array([samps[:, 0] <= px, samps[:, 1] <= py]) * 1
	actual_rho = np.corrcoef(samps_thresh[0], samps_thresh[1])[0, 1]
	print("Actual RHO: ", actual_rho)
	actual_scc = scc(samps_thresh[0], samps_thresh[1])
	print("Actual SCC: ", actual_scc)

	#print(f"Sample Pearson correlation coefficient: {corr_coef:.4f}")
	#plt.scatter(samps[:, 0], samps[:, 1])
	#plt.show()

	#Q = get_Q(2, lsb='right')
	#Q_inv = np.linalg.inv(Q)
	#p = np.array([1, x, y, cxy])
	#v = Q_inv @ p
	#P, C = get_C_from_v(v, return_P=True, pearson=False)
	#print(P)
	#print(C)

def xor_and_copula_vs_sim(num_samples=100, w=10):
	circ = XOR_with_AND()
	Mf = circ.get_PTM(lsb='left')
	ds = dataset_stack(np.array([0.75, 0.25]), num_samples).merge(dataset_sweep_1d(num_samples, start=1e-6, end=1-1e-6))

	Cins = [0.5]
	sim_results = []
	copula_outputs_gauss = []
	copula_outputs_frechet = []
	#Cin = np.array([[1, C, C],[C, 1, C],[C, C, 1]])
	Cin = np.array([
		[1, 0.5, 0.5],
		[0.5, 1, 0],
		[0.5, 0, 1]
	])
	for C in Cins:
		#sng = NONINT_LFSR_SNG(w, C, 3)
		sng = GAUSSIAN_COPULA_SNG(Cin)

		sim_result = sim_circ(sng, circ, ds, loop_print=True, Nset=2**14, compute_correct=False)
		sim_results.append(sim_result.out)
		copula_results_gauss = []
		copula_results_frechet = []
		for i, xs in enumerate(ds):
			print("{}/{}".format(i, ds.num))
			vin_gauss = get_DV_via_gaussian_copula(Cin, xs, verbose=False, lsb='left')
			vout_gauss = Mf.T @ vin_gauss
			P_gauss, _ = get_C_from_v(vout_gauss, return_P=True, lsb='left')
			copula_results_gauss.append(P_gauss[0])
			
			vin_pos = get_vin_mc1(xs)
			vin_uncorr = get_vin_mc0(xs, lsb='left')
			vin_frechet = C * vin_pos + (1 - C) * vin_uncorr
			vout_frechet = Mf.T @ vin_frechet
			P_frechet, _ = get_C_from_v(vout_frechet, return_P=True, lsb='left')
			copula_results_frechet.append(P_frechet[0])

			print("vin_gauss: ", vin_gauss)
			print("vin_frechet: ", vin_frechet)
			pass
			#print(get_C_from_v(vin, return_P=True, lsb='right'))
		copula_results_gauss = np.array(copula_results_gauss)
		copula_outputs_gauss.append(copula_results_gauss)
		copula_results_frechet = np.array(copula_results_frechet)
		copula_outputs_frechet.append(copula_results_frechet)

	import matplotlib.pyplot as plt

	# Plot simulated and copula model outputs for visual comparison
	plt.figure(figsize=(10, 5))
	for idx, sim_out in enumerate(sim_results):
		#plt.scatter(sim_out, copula_outputs[idx], label=f'Simulated Output (Cin={Cins[idx]})')
		plt.plot(np.linspace(0, 1, 100), sim_out, label=f'Simulated Output')
		plt.plot(np.linspace(0, 1, 100), copula_outputs_gauss[idx], label=f'Gaussian Copula Output')
		#plt.plot(np.linspace(0, 1, 100), copula_outputs_frechet[idx], label=f'Frechet Copula Output')
	plt.xlim(0, 1)
	plt.xlabel(r'$P_{X3}$ value')
	plt.ylabel(r'$P_Z$ value')
	plt.legend()
	plt.title('Simulated vs Copula Model Outputs')
	plt.grid(True)
	plt.show()

def sobel_copula_cameraman():
	"""Sweep the x-coordinate of the Sobel circuit and plot the correlation matrix."""
	
	w=6
	Cin_mat = np.ones((9, 9))
	circ = C_Sobel(Cin_mat, use_maj=False)
	ds = dataset_cameraman(3)
	#ds = dataset_uniform(500, 9)

	# --- Bitstream simulation ---
	time_start = time.time()
	print("Running bitstream simulation...")
	sng = LFSR_SNG(w, Cin_mat, nc=3)
	sim_result = sim_circ(sng, circ, ds, loop_print=True)
	time_end = time.time()
	print("Bitstream simulation time: ", time_end - time_start)
	#ds.disp_output_img(sim_result.out, 0)

	#Gaussian copula model with dependence analysis
	time_start = time.time()
	print("Running Gaussian copula model with dependence analysis")
	dependence_analysis = []
	sccs = []
	for i, xs in enumerate(ds):
		if i % 50 == 0:
			print("{}/{}".format(i, ds.num))
		#required overlap probs:
		z0z6 = np.minimum(xs[0], xs[6]) #get_vin_nonint_pair(Cin_mat[0, 6], xs[0], xs[6])[3] 
		z1z7 = np.minimum(xs[1], xs[7]) #get_vin_nonint_pair(Cin_mat[1, 7], xs[1], xs[7])[3]
		z2z8 = np.minimum(xs[2], xs[8]) #get_vin_nonint_pair(Cin_mat[2, 8], xs[2], xs[8])[3]
		z0z2 = np.minimum(xs[0], xs[2]) #get_vin_nonint_pair(Cin_mat[0, 2], xs[0], xs[2])[3]
		z3z5 = np.minimum(xs[3], xs[5]) #get_vin_nonint_pair(Cin_mat[3, 5], xs[3], xs[5])[3]
		z6z8 = np.minimum(xs[6], xs[8]) #get_vin_nonint_pair(Cin_mat[6, 8], xs[6], xs[8])[3]

		#w layer
		w12 = 0.25 * (z0z6 + 2*z1z7 + z2z8)
		w34 = 0.25 * (z0z2 + 2*z3z5 + z6z8)
		w1 = 0.25 * (xs[0] + 2*xs[1] + xs[2])
		w2 = 0.25 * (xs[6] + 2*xs[7] + xs[8])
		w3 = 0.25 * (xs[0] + 2*xs[3] + xs[6])
		w4 = 0.25 * (xs[2] + 2*xs[5] + xs[8])

		f = 0.5 * (w1 + w2 - 2*w12 + w3 + w4 - 2*w34)
		sccs.append(scc_prob(w1, w2, w12))
		dependence_analysis.append(f)
	dependence_analysis = np.array(dependence_analysis)
	time_end = time.time()
	print("Gaussian copula model with dependence analysis time: ", time_end - time_start)

	sccs = np.array(sccs)
	print("Average SCC: ", sccs.mean())

	# Ground truth output via .correct (analytical)
	correct_out = np.array([circ.correct(xs) for xs in ds])

	# Bitstream simulation output
	bitstream_out = sim_result.out

	# Copula model output
	copula_out = dependence_analysis

	# Compute RMSEs
	rmse_copula_vs_correct = np.sqrt(np.mean((copula_out - correct_out) ** 2))
	rmse_sim_vs_correct = np.sqrt(np.mean((bitstream_out - correct_out) ** 2))
	rmse_sim_vs_copula = np.sqrt(np.mean((bitstream_out - copula_out) ** 2))
	
	print(f"RMSE (Copula model vs Correct): {rmse_copula_vs_correct:.6g}")
	print(f"RMSE (Simulation vs Correct):   {rmse_sim_vs_correct:.6g}")
	print(f"RMSE (Simulation vs Copula model): {rmse_sim_vs_copula:.6g}")

	pass

	#ds.disp_output_img(dependence_analysis, 0)
	#ds.disp_output_img(gen_correct(circ, ds), 0)

def sobel_copula_x_sweep(num_samples=100, w=10):
	"""Sweep the x-coordinate of the Sobel circuit and plot the correlation matrix."""
	
	Cin_mat = np.ones((9, 9))
	circ = C_Sobel(Cin_mat, use_maj=False)
	ds = dataset_stack(np.array([0.25, 0.25, 0.75, 0.75, 0.25, 0.25, 0.25, 0.75]), num_samples).merge(dataset_sweep_1d(num_samples))

	# --- Bitstream simulation ---
	print("Running bitstream simulation...")
	sng = LFSR_SNG(w, Cin_mat, nc=3)
	sim_result = sim_circ(sng, circ, ds, loop_print=False)
	correct_vals = sim_result.correct

	#Gaussian copula model with dependence analysis
	print("Running Gaussian copula model with dependence analysis")
	dependence_analysis = []
	for i, xs in enumerate(ds):
		#required overlap probs:
		z0z6 = get_vin_nonint_pair(Cin_mat[0, 6], xs[0], xs[6])[3] 
		z1z7 = get_vin_nonint_pair(Cin_mat[1, 7], xs[1], xs[7])[3]
		z2z8 = get_vin_nonint_pair(Cin_mat[2, 8], xs[2], xs[8])[3]
		z0z2 = get_vin_nonint_pair(Cin_mat[0, 2], xs[0], xs[2])[3]
		z3z5 = get_vin_nonint_pair(Cin_mat[3, 5], xs[3], xs[5])[3]
		z6z8 = get_vin_nonint_pair(Cin_mat[6, 8], xs[6], xs[8])[3]

		#w layer
		w12 = 0.25 * (z0z6 + 2*z1z7 + z2z8)
		w34 = 0.25 * (z0z2 + 2*z3z5 + z6z8)
		w1 = 0.25 * (xs[0] + 2*xs[1] + xs[2])
		w2 = 0.25 * (xs[6] + 2*xs[7] + xs[8])
		w3 = 0.25 * (xs[0] + 2*xs[3] + xs[6])
		w4 = 0.25 * (xs[2] + 2*xs[5] + xs[8])

		f = 0.5 * (w1 + w2 - 2*w12 + w3 + w4 - 2*w34)
		dependence_analysis.append(f)
	dependence_analysis = np.array(dependence_analysis)

	xs = np.linspace(0, 1, num_samples)
	plt.plot(xs, dependence_analysis, label='Copula model')
	plt.plot(xs, correct_vals, label='Correct')
	plt.plot(xs, sim_result.out, label='Simulation')
	plt.legend()
	plt.show()

def sobel_copula_vs_sim(num_samples=100, w=10):
	"""Compare Gaussian copula model against bitstream simulation for the Sobel circuit.

	Runs three analyses on the same dataset of random 3x3 pixel windows:
	  1. Bitstream simulation (LFSR-based SNG)
	  2. Frechet PTV model (sim_circ_PTM)
	  3. Gaussian copula model (get_vin_via_gaussian_copula + PTM propagation)

	Produces scatter plots and RMSE comparisons.
	"""

	Cin_mat = np.ones((9, 9))
	
	circ = C_Sobel(Cin_mat, use_maj=False)
	ds = dataset_uniform(num_samples, 9)

	# --- Bitstream simulation of just the MUX layer of the Sobel filter --- 
	print("Running bitstream simulation of just the MUX layer of the Sobel filter...")
	sobel_muxes = C_SobelMuxes()
	sng = LFSR_SNG(w, Cin_mat, nc=3)
	sim_result = sim_circ(sng, sobel_muxes, ds, compute_correct=False, loop_print=False)
	sobel_muxes_Cs = sim_result.Cs
	sobel_muxes_C_avg = sobel_muxes_Cs.mean(axis=0)
	sobel_muxes_C_avg *= np.array([
		[1, 1, 0, 0],
		[1, 1, 0, 0],
		[0, 0, 1, 1],
		[0, 0, 1, 1],
	])
	print(sobel_muxes_C_avg)

	# --- Bitstream simulation ---
	print("Running bitstream simulation...")
	sng = LFSR_SNG(w, Cin_mat, nc=3)
	sim_result = sim_circ(sng, circ, ds, loop_print=False)

	# --- Frechet PTV model ---
	print("Running Frechet PTV model...")
	ptm_result = sim_circ_PTM(circ, ds, Cin_mat)
	ptm_outputs = ptm_result.out.flatten()

	# --- Gaussian copula model ---
	print("Running Gaussian copula model...")
	Mf = circ.get_PTM(lsb='right')
	v_consts = get_PTV(np.identity(3), np.array([0.5, 0.5, 0.5]), lsb='right')
	copula_outputs = []
	for i, xs in enumerate(ds):
		if i % 50 == 0:
			print(f"  Copula sample {i}/{num_samples}")
		vin = get_DV_via_gaussian_copula(Cin_mat, xs, verbose=False)
		vin_full = np.kron(vin, v_consts)
		vout = Mf.T @ vin_full
		P, _ = get_C_from_v(vout, return_P=True, lsb='right')
		copula_outputs.append(P[0])
	copula_outputs = np.array(copula_outputs)

	#Gaussian copula model with dependence analysis
	print("Running Gaussian copula model with dependence analysis")
	dependence_analysis = []
	for i, xs in enumerate(ds):
		#required overlap probs:
		z0z6 = get_vin_nonint_pair(Cin_mat[0, 6], xs[0], xs[6])[3] 
		z1z7 = get_vin_nonint_pair(Cin_mat[1, 7], xs[1], xs[7])[3]
		z2z8 = get_vin_nonint_pair(Cin_mat[2, 8], xs[2], xs[8])[3]
		z0z2 = get_vin_nonint_pair(Cin_mat[0, 2], xs[0], xs[2])[3]
		z3z5 = get_vin_nonint_pair(Cin_mat[3, 5], xs[3], xs[5])[3]
		z6z8 = get_vin_nonint_pair(Cin_mat[6, 8], xs[6], xs[8])[3]

		#w layer
		w12 = 0.25 * (z0z6 + 2*z1z7 + z2z8)
		w34 = 0.25 * (z0z2 + 2*z3z5 + z6z8)
		w1 = 0.25 * (xs[0] + 2*xs[1] + xs[2])
		w2 = 0.25 * (xs[6] + 2*xs[7] + xs[8])
		w3 = 0.25 * (xs[0] + 2*xs[3] + xs[6])
		w4 = 0.25 * (xs[2] + 2*xs[5] + xs[8])

		f = 0.5 * (w1 + w2 - 2*w12 + w3 + w4 - 2*w34)
		dependence_analysis.append(f)
	dependence_analyis = np.array(dependence_analysis)

	# --- Gaussian copula model on MUX stage only
	print("Running Gaussian copula model on RCED stage...")
	rced = C_RCED()
	Mf = rced.get_PTM(lsb='right')
	v_consts = get_PTV(np.identity(1), np.array([0.5]), lsb='right')
	copula_outputs_rced = []
	for i, xs in enumerate(ds):
		if i % 50 == 0:
			print(f"  Copula sample {i}/{num_samples}")
		vin = get_DV_via_gaussian_copula(sobel_muxes_C_avg, xs, verbose=False)
		vin_full = np.kron(vin, v_consts)
		vout = Mf.T @ vin_full
		P, _ = get_C_from_v(vout, return_P=True, lsb='right')
		copula_outputs_rced.append(P[0])
	copula_outputs_rced = np.array(copula_outputs_rced)

	# --- Comparison ---
	correct_vals = sim_result.correct

	sim_rmse = np.sqrt(np.mean((sim_result.out - correct_vals) ** 2))
	ptm_rmse = np.sqrt(np.mean((ptm_outputs - correct_vals) ** 2))
	copula_rmse = np.sqrt(np.mean((copula_outputs - correct_vals) ** 2))
	copula_vs_sim = np.sqrt(np.mean((copula_outputs - sim_result.out) ** 2))
	copula_vs_ptm = np.sqrt(np.mean((copula_outputs - ptm_outputs) ** 2))
	copula_vs_rced = np.sqrt(np.mean((copula_outputs_rced - sim_result.out) ** 2))

	print(f"\n--- RMSE Results ---")
	print(f"Simulation  vs Correct:   {sim_rmse:.6f}")
	print(f"Frechet PTV vs Correct:   {ptm_rmse:.6f}")
	print(f"Copula      vs Correct:   {copula_rmse:.6f}")
	print(f"Copula      vs Simulation:{copula_vs_sim:.6f}")
	print(f"Copula      vs Frechet:   {copula_vs_ptm:.6f}")
	print(f"Copula      vs RCED:      {copula_vs_rced:.6f}")

	# --- Plots ---
	fig, axes = plt.subplots(1, 6, figsize=(15, 5))
	vmax = max(correct_vals.max(), sim_result.out.max(), copula_outputs.max()) * 1.05

	axes[0].scatter(correct_vals, sim_result.out, s=8, alpha=0.5, label='Simulation')
	axes[0].plot([0, vmax], [0, vmax], 'r--', lw=1)
	axes[0].set_xlabel('Correct (Sobel)')
	axes[0].set_ylabel('Simulation Output')
	axes[0].set_title(f'Simulation vs Correct\nRMSE = {sim_rmse:.4f}')
	axes[0].set_aspect('equal')
	axes[0].set_xlim(0, vmax)
	axes[0].set_ylim(0, vmax)

	axes[1].scatter(correct_vals, copula_outputs, s=8, alpha=0.5, color='tab:orange', label='Copula')
	axes[1].scatter(correct_vals, ptm_outputs, s=8, alpha=0.5, color='tab:green', label='Frechet PTV')
	axes[1].plot([0, vmax], [0, vmax], 'r--', lw=1)
	axes[1].set_xlabel('Correct (Sobel)')
	axes[1].set_ylabel('Model Output')
	axes[1].set_title(f'Copula & Frechet vs Correct\nCopula RMSE={copula_rmse:.4f}, PTV RMSE={ptm_rmse:.4f}')
	axes[1].legend(markerscale=2)
	axes[1].set_aspect('equal')
	axes[1].set_xlim(0, vmax)
	axes[1].set_ylim(0, vmax)

	axes[2].scatter(sim_result.out, copula_outputs, s=8, alpha=0.5, color='tab:purple')
	axes[2].plot([0, vmax], [0, vmax], 'r--', lw=1)
	axes[2].set_xlabel('Simulation Output')
	axes[2].set_ylabel('Copula Model Output')
	axes[2].set_title(f'Gaussian copula vs Simulation\nRMSE = {copula_vs_sim:.4f}')
	axes[2].set_aspect('equal')
	axes[2].set_xlim(0, vmax)
	axes[2].set_ylim(0, vmax)

	axes[3].scatter(sim_result.out, ptm_outputs, s=8, alpha=0.5, color='tab:green')
	axes[3].plot([0, vmax], [0, vmax], 'r--', lw=1)
	axes[3].set_xlabel('Simulation Output')
	axes[3].set_ylabel('Frechet PTV Model Output')
	axes[3].set_title(f'Frechet copula vs Simulation\nRMSE = {np.sqrt(np.mean((ptm_outputs - sim_result.out) ** 2)):.4f}')
	axes[3].set_aspect('equal')
	axes[3].set_xlim(0, vmax)
	axes[3].set_ylim(0, vmax)

	axes[4].scatter(sim_result.out, copula_outputs_rced, s=8, alpha=0.5, color='tab:purple')
	axes[4].plot([0, vmax], [0, vmax], 'r--', lw=1)
	axes[4].set_xlabel('Simulation Output')
	axes[4].set_ylabel('Copula Model on RCED Only')
	axes[4].set_title(f'Copula vs Simulation\nRMSE = {copula_vs_sim:.4f}')
	axes[4].set_aspect('equal')
	axes[4].set_xlim(0, vmax)
	axes[4].set_ylim(0, vmax)

	axes[5].scatter(sim_result.out, dependence_analyis, s=8, alpha=0.5, color='tab:purple')
	axes[5].plot([0, vmax], [0, vmax], 'r--', lw=1)
	axes[5].set_xlabel('Simulation Output')
	axes[5].set_ylabel('Dependence analysis')
	axes[5].set_title(f'Dependence analysis vs Simulation\nRMSE = {copula_vs_sim:.4f}')
	axes[5].set_aspect('equal')
	axes[5].set_xlim(0, vmax)
	axes[5].set_ylim(0, vmax)

	plt.suptitle('GBED Circuit: Gaussian Copula Model vs Simulation', fontsize=14)
	plt.tight_layout()
	plt.show()

def plot_SCC_to_Pearson():
    """
    Plots the relationship between Spearman's correlation coefficient (SCC) and
    the corresponding Pearson correlation, for five different (px, py) pairs.
    One of them is (0.5, 0.5).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    px_py_list = [
        (0.5, 0.5),
        (0.2, 0.8),
        (0.8, 0.2),
        (0.3, 0.6),
        (0.7, 0.7)
    ]
    colors = ['b', 'g', 'r', 'c', 'm']

    scc_values = np.linspace(-1, 1, 500)
    plt.figure(figsize=(8, 6))
    for (px, py), color in zip(px_py_list, colors):
        pearson_corrs = [SCC_to_Pearson(scc, px, py) for scc in scc_values]
        label = f"px={px:.2f}, py={py:.2f}"
        plt.plot(scc_values, pearson_corrs, label=label, color=color)

    plt.xlabel("SCC")
    plt.ylabel("Pearson correlation")
    plt.title("Pearson Correlation vs SCC (various px, py)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

"""Below is code for fitting a Gaussian copula to an SCC matrix jointly instead of using pairwise SCC values"""
def scc_from_joint(p_i, p_j, p_ij):
    independent = p_i * p_j

    if p_ij >= independent:
        denom = min(p_i, p_j) - independent
    else:
        denom = independent - max(p_i + p_j - 1.0, 0.0)

    if abs(denom) < 1e-15:
        return 0.0

    return (p_ij - independent) / denom


def gaussian_scc_from_rho(p_i, p_j, rho):
    rho = np.clip(rho, -0.999999, 0.999999)

    t_i = norm.ppf(p_i)
    t_j = norm.ppf(p_j)

    cov = np.array([
        [1.0, rho],
        [rho, 1.0],
    ])

    p_ij = multivariate_normal.cdf(
        [t_i, t_j],
        mean=[0.0, 0.0],
        cov=cov,
    )

    return scc_from_joint(p_i, p_j, p_ij)


def unpack_B(x, n, rank):
    B = x.reshape(n, rank)

    norms = np.linalg.norm(B, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)

    return B / norms


def fit_gaussian_copula_corr(C_target, p, rank=None, num_restarts=10):
    """
    Fit a valid Gaussian copula correlation matrix R directly.

    Parameters
    ----------
    C_target : np.ndarray
        Target SCC matrix.
    p : np.ndarray
        Marginal probabilities P(X_i = 1).
    rank : int or None
        Dimension of Gram vectors. Use n for full flexibility.
    num_restarts : int
        Number of random initializations.

    Returns
    -------
    R_best : np.ndarray
        PSD latent Pearson correlation matrix with unit diagonal.
    result_best : scipy.optimize.OptimizeResult
        Best optimization result.
    """

    C_target = np.asarray(C_target, dtype=float)
    p = np.asarray(p, dtype=float)

    n = C_target.shape[0]
    if rank is None:
        rank = n

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def objective(x):
        B = unpack_B(x, n, rank)
        R = B @ B.T

        err = 0.0
        for i, j in pairs:
            c_hat = gaussian_scc_from_rho(p[i], p[j], R[i, j])
            diff = c_hat - C_target[i, j]
            err += diff * diff

        return err

    best_result = None
    best_value = np.inf

    rng = np.random.default_rng(0)

    for _ in range(num_restarts):
        x0 = rng.normal(size=(n, rank)).ravel()

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={
                "maxiter": 1000,
                "ftol": 1e-12,
            },
        )

        if result.fun < best_value:
            best_result = result
            best_value = result.fun

    B_best = unpack_B(best_result.x, n, rank)
    R_best = B_best @ B_best.T

    R_best = 0.5 * (R_best + R_best.T)
    np.fill_diagonal(R_best, 1.0)

    return R_best, best_result


