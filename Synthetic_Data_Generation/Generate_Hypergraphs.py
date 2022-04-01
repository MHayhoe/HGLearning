import numpy as np
import pickle
import torch
from Utils import *
from Hypergraphs import *
from Source_Localization import hypergraphSources

if __name__ == '__main__':
    H = Hypergraph([[1, 2, 3], [2, 3, 4], [4, 5], [1, 6, 7, 8]])
    # H = Hypergraph([[1, 2, 3], [2, 3, 4], [4, 5]])
    # x = np.random.rand(H.N)*2-1
    x = np.array([0, 0, 1., 0, 0, 0, 0, 0])
    nTrain = 5
    nValid = 3
    nTest = 2
    sourceHyperedges = [0, 1]
    num_steps = 20
    useGPU = True

    GSOs = [H.clique_laplacian(), H.line_laplacian()]
    incidence_matrix = H.incidence_matrix()
    data = hypergraphSources(H, nTrain, nValid, nTest, sourceHyperedges, tMax=num_steps, dataType=torch.float64,
                             device='cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu')
    with open('../Learning/data/sourceLoc/sourceLoc_GSOs.pkl','wb') as f:
        pickle.dump(GSOs, f)
    with open('../Learning/data/sourceLoc/sourceLoc_incidence_matrices.pkl', 'wb') as f:
        pickle.dump(incidence_matrix, f)
    with open('../Learning/data/sourceLoc/sourceLoc_data.pkl', 'wb') as f:
        pickle.dump(data, f)
