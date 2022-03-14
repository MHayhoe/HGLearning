import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import networkx as nx
from tqdm import tqdm


class Hypergraph:
    def __init__(self, hyperedges=[]):
        self.hyperedges = list(map(lambda hedge: tuple(sorted(hedge)), hyperedges))
        self.laplacian = self.laplacian_operator()

    # Computes the energy of the signal x on the hypergraph
    def energy_fn(self, x):
        energy = 0
        for hedge in self.hyperedges:
            energy += np.max((x[hedge] - x[hedge])**2)
        return energy

    # Returns the Laplacian operator, i.e., the gradient of the energy
    def laplacian_operator(self):
        return jit(grad(self.energy_fn))
