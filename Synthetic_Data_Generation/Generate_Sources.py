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
    n_points = 100  # number of datapoints
    epsilon = 1.2  # max distance between datapoints to draw a simplex between
    noise = 0  # noise to add during drawing of samples

    # Draws datapoints from a torus
    points = tadasets.torus(n_points, c=2, a=1, noise=noise)
    # Builds a Cech Complex (saving only the largest connected component)
    SC = CechComplex(points, epsilon, lcc = True)

    # Plots SC in new window
    # plot_2d_sc(SC)

    # Create a hypergraph from the Cech complex
    H = from_SC(SC)

    # Plots hypergraph in new window
    # plot_2d_hg(H)

    # Now, generate the samples for the source localization problem
    nTrain = 500
    nValid = 300
    nTest = 200
    # Only treat the first 10% of hyperedges as possible sources
    sourceHyperedges = np.arange(np.round(H.M / 10)).astype(np.int8)
    num_steps = 10
    useGPU = True

    # Generate the GSOs for the clique and line expansion
    print('Generating GSOs...')
    GSOs = [H.clique_laplacian(), H.line_laplacian()]

    # Compute the incidence matrix
    print('Creating incidence matrix...')
    incidence_matrix = [H.incidence_matrix()]

    # Create the samples for source localization
    print('Generating sources.')
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
