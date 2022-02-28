from Synthetic_Complexes import *
from Generate_Complexes import *
from Utils import *

if __name__=="__main__":
    n_dim = 2
    boundary_width = 1
    n_points = 200
    epsilon = 0.1
    SC = generate_complex(n_dim, boundary_width, n_points, epsilon)
    if n_dim == 2:
        plot_2d_sc(SC)
