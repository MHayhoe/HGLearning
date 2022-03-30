import numpy as np

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

    data = hypergraphSources(H, nTrain, nValid, nTest, sourceHyperedges, tMax=num_steps)
    print(data.samples)
    # plot_diffusion(H, x, num_steps)
