import numpy as np
from Synthetic_Complexes import *
import tadasets

def generate_complex(n_points, epsilon, n_dim = None, boundary_width = None, noise = 0.1, seed = None):
    #points = np.random.uniform(low = 0, high = boundary_width, size = (n_points, n_dim))
    points = tadasets.torus(n_points, c = 2, a = 1, noise = noise)

    return CechComplex(points, epsilon, lcc = True)