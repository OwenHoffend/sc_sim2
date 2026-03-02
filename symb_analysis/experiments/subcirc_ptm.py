#Experiments relating to subcircuit PTM analysis

import numpy as np
import sympy as sp
from sim.PTM import get_func_mat
from sim.PTV import copula_transform

def and_or_example(x, y):
	return (x & y), (x | y)

def test_and_or_example():
	ptm = get_func_mat(and_or_example, 2, 2)
	print(ptm * 1)
	print(copula_transform(ptm))

def armin_example_6(x1, x2, x3, x4, x5):
	#Example from Fig. 6 in 
	# Alaghi, A., & Hayes, J. (2015). 
	# Dimension reduction in statistical simulation of digital circuits.
	return (x1 & x2 & x4) | (x1 & x3 & ~x4 & ~x5) | (x2 & x3 & ~x4 & x5)

def test_armin_example_6():
	ptm = get_func_mat(armin_example_6, 5, 1)
	print(ptm * 1)
	print(copula_transform(ptm))