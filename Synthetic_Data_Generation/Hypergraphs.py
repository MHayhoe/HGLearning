import jax.numpy as jnp
from jax import grad, jit
import networkx as nx
from tqdm import tqdm
from Synthetic_Complexes import *

class Hypergraph:
    def __init__(self, hyperedges=[]):
        self.N = 0
        self.node_map = {}
        self.hyperedges = list(map(lambda hedge: sorted([self.map_node(v) for v in hedge]), hyperedges))
        self.laplacian = self.laplacian_operator()

    # Maps nodes. If it's been seen before, this assigns the old value, otherwise increments node count.
    # node_map is the same as a defaultdict object, but faster with large datasets.
    def map_node(self, v):
        if v not in self.node_map:
            self.node_map[v] = self.N
            self.N += 1
        return self.node_map[v]

    # Computes the energy of the signal x on the hypergraph
    def energy_fn(self, x):
        energy = 0
        for hedge in self.hyperedges:
            hedge = jnp.array(hedge)
            energy += jnp.max((x[hedge, jnp.newaxis] - x[hedge])**2) / hedge.size
        return energy

    # Returns the Laplacian operator, i.e., half the gradient of the energy
    def laplacian_operator(self):
        return lambda x: 1/2*jit(grad(self.energy_fn))(x)

    # Diffuses the signal x according to L for k times, i.e., x(t+1) = x(t) - L(x(t)).
    def diffuse(self, x, k=1):
        for t in range(k):
            x = x - self.laplacian(x)

        return x

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
