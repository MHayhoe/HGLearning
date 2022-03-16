from Synthetic_Complexes import *
from Generate_Complexes import *
from Source_Localization import *
from Utils import *
from Hypergraphs import *

if __name__=="__main__":
    h = Hypergraph([(1,2,3),(2,3,4,5),(4,7),(5,6),(3,5)])
    hd = h.sc_dual()
    print(hd.simplices)
    print(hd.boundary_maps())
    print(hd.hodge_laps)

    
    shape = 'torus' # unit or torus
    n_points = 1000
    epsilon = 0.4
    seed = 0

    if shape == 'unit':
        n_dim = 2
        boundary_width = 1
        noise = None
    else:
        n_dim = None
        boundary_width = None
        noise = 0
    
    SC = generate_complex(n_points, epsilon, n_dim = n_dim, boundary_width = boundary_width, noise = noise, seed = seed)

    # print(generate_hypergraph_diffusion(SC, 2000, 20, 10, 5))

    plot_2d_sc(SC)
    
