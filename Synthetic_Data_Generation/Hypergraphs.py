import jax.numpy as jnp
from jax import grad, jit
import networkx as nx
from Simplicial_Complexes import *

class Hypergraph:
    def __init__(self, hyperedges=[], signals=None):
        self.N = 0
        self.node_map = {}
        self.hyperedges = list(map(lambda hedge: sorted([self.map_node(v) for v in hedge]), hyperedges))
        self.M = len(self.hyperedges)
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

    # Checks the shape of signals to see if they match hypergraph cardinality
    def check_shape_signals(self, x):
        return x.shape[0] == self.N

    # Diffuses the signal x according to L for k times, i.e., x(t+1) = x(t) - L(x(t)).
    def diffuse(self, x=None, k=1):
        for _ in range(k):
            x = x - self.laplacian(x)

        return x

    # Computes the Simplicial Complex from the dual of the hypergraph
    def sc_dual(self, x):
        # First, constructs a simple graph, with hyperedges as nodes and edges between hnodes
        # if they share a node in the hypergraph
        hyperedge_tuples = [tuple(hedge) for hedge in self.hyperedges]

        g = nx.Graph()
        g.add_nodes_from(hyperedge_tuples)

        for hedge1 in range(self.M - 1):
            for hedge2 in range(hedge1+1, self.M):
                if len(set(hyperedge_tuples[hedge1]) & set(hyperedge_tuples[hedge2])) > 0:
                    g.add_edge(hyperedge_tuples[hedge1], hyperedge_tuples[hedge2])
        
        # Next, find cliques in this graph
        simplices = map(tuple, list(nx.find_cliques(g)))

        # Create Simplicial Complex from cliques
        SC = SimplicialComplex(simplices)
        if len(x.shape) == 1:
            x = np.reshape(x, (-1, 1))

        # Get the faces to compute signals
        SC_signals = []
        max_order = max(map(len, SC.face_set))
        for order in range(max_order):
            faces = SC.n_faces(order)
            sig_order = np.zeros((len(faces), x.shape[1]))
            for i, face in enumerate(faces):
                common_nodes = list(set.intersection(*map(set, face)))
                sig_order[i, :] = np.mean(x[common_nodes, :], axis=0)
            SC_signals.append(sig_order)

        return SC, SC_signals
