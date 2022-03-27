from Utils import *
from Hypergraphs import *
from Simplicial_Complexes import *
import numpy as np
import tadasets


if __name__ == '__main__':
    ### Example of moving from Hypergraph -> Simplicial Complex 
    H = Hypergraph([[1, 2, 3], [2, 3, 4], [4, 5],[1,6,7,8]])
    x = np.random.rand(H.N) * 2 - 1  
    num_steps = 5

    plot_diffusion(H, x, num_steps) # Hypergraph Diffusion

    h = Hypergraph([(1,2,3),(2,3,4,5),(4,7),(5,6),(3,5)])
    sc, sc_signal = h.sc_dual(x)
    print(sc.simplices)
    print(sc.boundary_maps())
    print(sc.hodge_laps)
    ds = sc.diffuse(sc_signal, 5)

    ### Example of generating a simplicial complex from a torus
    n_points = 1000 # number of datapoints
    epsilon = 0.4 # max distance between datapoints to draw a simplex between
    noise = 0 # noise to add during drawing of samples

    # Draws datapoints from a torus
    points = tadasets.torus(n_points, c = 2, a = 1, noise = noise)
    # Builds a Cech Complex
    SC = CechComplex(points, epsilon, lcc = True)

    # Plots SC in new window
    plot_2d_sc(SC)