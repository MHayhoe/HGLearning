import numpy as np
import pickle
import torch
from Utils import *
from Source_Localization import hypergraphSources
from Simplicial_Complexes import *
from Hypergraphs import *
import numpy as np
import tadasets

if __name__ == '__main__':
    # First, we generate a Cech Complex by sampling points from a torus
    n_points = 500  # number of datapoints
    epsilon = 0.4  # max distance between datapoints to draw a simplex between
    noise = 0.1  # noise to add during drawing of samples
    nTrain = 500    # number of training samples to generate
    nValid = 300    # number of validation samples to generate
    nTest = 1000     # number of testing samples to generate
    num_steps = 25  # number of steps for diffusion
    useGPU = True   # whether to use the GPU for generating samples (for gradient of HG energy function)

    # Draws datapoints from a torus
    points = tadasets.torus(n_points, c=2, a=1, noise=noise)
    # Builds a Cech Complex (saving only the largest connected component)
    SC = CechComplex(points, epsilon, lcc=True)

    # Plots SC in new window
    plot_2d_sc(SC, '../Learning/data/sourceLoc/')

    # Create a hypergraph from the Cech complex
    H = from_SC(SC)
    print('Hypergraph has {} nodes and {} hyperedges.'.format(H.N, H.M))
    # Treat 5 hyperedges at random as possible sources
    numSources = 5
    sourceHyperedges = np.random.choice(np.arange(H.M), size=numSources, replace=False)

    # Plots hypergraph in new window
    print('Plotting...')
    plot_diffusions_hg(SC, H, sourceHyperedges, num_steps, '../Learning/data/sourceLoc/')
    plot_2d_hg(H, None, sourceHyperedges, '../Learning/data/sourceLoc/')

    # Generate the GSOs for the clique and line expansion
    print('Generating GSOs...')
    GSOs = [H.clique_laplacian(), H.line_laplacian()]

    # Compute the incidence matrix
    print('Creating incidence matrix...')
    incidence_matrix = [H.incidence_matrix()]

    # Create the samples for source localization
    print('Generating {} sources...'.format(numSources))
    data = hypergraphSources(H, nTrain, nValid, nTest, sourceHyperedges, tMax=num_steps, dataType=torch.float64,
                             device='cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu')

    # Save everything
    print('Saving...')
    with open('../Learning/data/sourceLoc/sourceLoc_GSOs.pkl','wb') as f:
        pickle.dump(GSOs, f)
    with open('../Learning/data/sourceLoc/sourceLoc_incidence_matrices.pkl', 'wb') as f:
        pickle.dump(incidence_matrix, f)
    with open('../Learning/data/sourceLoc/sourceLoc_data.pkl', 'wb') as f:
        pickle.dump(data, f)
