import numpy as np
from statsmodels.distributions.copula.api import GaussianCopula
from statsmodels.distributions.copula.api import CopulaDistribution
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sim.PTV import get_C_from_v
from sim.PTV import get_Q
from sim.PTM import get_func_mat
from symb_analysis.experiments.subcirc_ptm import and_or_example
from sim.SCC import SCC_to_Pearson, Pearson_to_SCC
from scipy import stats, optimize
from sim.SCC import norm_inv
from sim.SCC import scc_prob
from sim.SCC import scc
from sim.visualization import plot_scc_heatmap

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
				lambda r: binary_corr_from_latent(r, px, py) - target_corr,
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

def binary_corr_from_latent(rho, px, py):
	a = stats.norm.ppf(px)
	b = stats.norm.ppf(py)
	pxy = stats.multivariate_normal.cdf([a, b], mean=[0,0],
								cov=[[1,rho],[rho,1]])
	#return (pxy - px*py) / np.sqrt(px*(1-px)*py*(1-py))
	return scc_prob(px, py, pxy)

def gauss_copula_test_2d():
	c = 0
	px = 0.75
	py = 0.25

	#target_corr = SCC_to_Pearson(c, px, py)
	target_corr = c



	import matplotlib.pyplot as plt
	import numpy as np

	# Plot binary_corr_from_latent as a function of rho
	rho_values = np.linspace(-0.999, 0.999, 200)
	corr_values = [binary_corr_from_latent(rho) for rho in rho_values]

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
		lambda r: binary_corr_from_latent(r) - target_corr,
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


