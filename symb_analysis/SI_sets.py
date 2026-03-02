from sim.PTV import copula_transform_matrix

def compute_SI_sets(Mf, lsb='right'):
	T = copula_transform_matrix(Mf, lsb=lsb)
	