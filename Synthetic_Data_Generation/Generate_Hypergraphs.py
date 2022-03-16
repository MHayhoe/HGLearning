import numpy as np

from Utils import *
from Hypergraphs import *


if __name__ == '__main__':
    H = Hypergraph([[1, 2, 3], [2, 3, 4], [4, 5],[1,6,7,8]])
    # H = Hypergraph([[1, 2, 3], [2, 3, 4], [4, 5]])
    x = np.random.rand(H.N)*2-1  # np.array([0,0,1.,0,0])
    num_steps = 5

    plot_diffusion(H, x, num_steps)
