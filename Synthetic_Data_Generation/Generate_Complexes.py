import numpy as np
from Synthetic_Complexes import *

def generate_complex(n_dim, boundary_width, n_points, epsilon):
    points = np.random.uniform(low = 0, high = boundary_width, size = (n_points, n_dim))
    print(CechComplex(points, epsilon, lcc = True).simplices)
    return CechComplex(points, epsilon, lcc = True)