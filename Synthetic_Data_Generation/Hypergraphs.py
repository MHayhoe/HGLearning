import jax.numpy as jnp
from jax import grad, jit
import networkx as nx
from tqdm import tqdm
from Synthetic_Complexes import *


class Hypergraph:
    def __init__(self, hyperedges=[]):
        self.hyperedges = list(map(lambda hedge: tuple(sorted(hedge)), hyperedges))
        self.laplacian = self.laplacian_operator()

    # Computes the energy of the signal x on the hypergraph
    def energy_fn(self, x):
        energy = 0
        for hedge in self.hyperedges:
            energy += jnp.max((x[hedge, jnp.newaxis] - x[hedge])**2)
        return energy

    # Returns the Laplacian operator, i.e., the gradient of the energy
    def laplacian_operator(self):
        return jit(grad(self.energy_fn))

    # Computes the Simplicial Complex from the dual of the hypergraph
    def sc_dual(self):
        # First, constructs a simple graph, with hyperedges as nodes and edges between hnodes
        # if they share a node in the hypergraph
        g = nx.Graph()
        g.add_nodes_from(self.hyperedges)

        for hedge1 in range(len(self.hyperedges)-1):
            for hedge2 in range(hedge1+1, len(self.hyperedges)):
                if len(set(self.hyperedges[hedge1]) & set(self.hyperedges[hedge2])) > 0:
                    g.add_edge(self.hyperedges[hedge1],self.hyperedges[hedge2])
        
        # Next, find cliques in this graph
        simplices = map(tuple, list(nx.find_cliques(g)))

        # Create Simplicial Complex from cliques
        return SimplicialComplex(simplices)