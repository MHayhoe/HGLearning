import numpy as np
import torch
import networkx as nx
from Simplicial_Complexes import SimplicialComplex
from tqdm import tqdm
import csv


# Computes a maximal hypergraph from the simplicial complex. In other words, takes all nodes from the SC,
# and adds hyperedges corresponding to the largest-order simplices in the SC (ignoring lower-order faces).
def from_SC(S : SimplicialComplex):
    return Hypergraph(S.simplices)


# Loads hyperedges from a CSV
def from_CSV(fname):
    hyperedge_list = []
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if len(row) <= 25:
                hyperedge_list.append(row)
    return Hypergraph(hyperedge_list)


# Creates a normalized hypergraph Laplacian matrix from a hypergraph incidence matrix
def HG_normalized_Laplacian_from_incidence(B_list):
    L_list = []

    for B in B_list:
        hedge_weights = torch.sum(B, axis=0)
        D_e_inv = torch.float_power(hedge_weights, -1)
        node_weights = torch.sum(B, axis=1)
        D_u_neghalf = torch.float_power(node_weights, -1/2)
        BD_u = (B.T * D_u_neghalf)
        L = torch.eye(B.shape[0], device=B.device) - BD_u.T * D_e_inv @ BD_u
        L_list.append(L)
    return L_list

class Hypergraph:
    def __init__(self, hyperedges=[], signals=None):
        self.N = 0
        self.node_map = {}
        self.hyperedges = list(map(lambda hedge: sorted([self.map_node(v) for v in hedge]), hyperedges))
        self.M = len(self.hyperedges)
        self.B = self.incidence_matrix()
        self.laplacian = self.laplacian_operator

    # Maps nodes. If it's been seen before, this assigns the old value, otherwise increments node count.
    # node_map is the same as a defaultdict object, but faster with large datasets.
    def map_node(self, v):
        if v not in self.node_map:
            self.node_map[v] = self.N
            self.N += 1
        return self.node_map[v]

    '''
    # The following code computes the Laplacian operator using the autodifferentiation package JAX.
    # There were several errors in how it computed the gradient, so this should not be used.
    # Computes the energy of the signal x on the hypergraph
    def energy_fn(self, x):
        energy = 0
        for hedge in self.hyperedges:
            hedge = jnp.array(hedge)
            energy += jnp.max((x[hedge, jnp.newaxis] - x[hedge])**2) / hedge.size
        return energy

    def energy_fn_matrix(self, x):
        E = self.B.T * x
        E = (E[:, :, np.newaxis] - E[:, np.newaxis, :])**2
        hedge_sizes = jnp.array([len(hedge) for hedge in self.hyperedges])
        return jnp.sum(jnp.max(jnp.max(E, axis=2), axis=1) / hedge_sizes)

    # Returns the Laplacian operator, i.e., half the gradient of the energy
    def laplacian_operator(self):
        return lambda x: 1/2*jit(grad(self.energy_fn_matrix))(x)
    '''

    # Checks the shape of signals to see if they match hypergraph cardinality
    def check_shape_signals(self, x):
        return x.shape[0] == self.N

    # Computes the Laplacian as the gradient of the energy function
    def laplacian_operator(self, x):
        xL = np.zeros(x.shape)
        for hedge in self.hyperedges:
            E = np.abs(x[hedge, np.newaxis] - x[hedge]) / len(hedge)
            maxVal = np.max(E)
            # Only do the rest if there's energy here
            if maxVal > 0:
                maxRows, maxCols = np.nonzero(E == maxVal)
                for i in range(len(maxRows)):
                    nodeRow = hedge[maxRows[i]]
                    nodeCol = hedge[maxCols[i]]
                    if x[nodeRow] > x[nodeCol]:
                        xL[nodeRow] = maxVal
                        xL[nodeCol] = -maxVal
                    else:
                        xL[nodeRow] = -maxVal
                        xL[nodeCol] = maxVal
        return xL

    # Diffuses the signal x according to L for k times, i.e., x(t+1) = x(t) - L(x(t)).
    def diffuse(self, x0=None, k=1):
        x = np.zeros((k+1, self.N))
        if x0 is not None:
            assert x0.shape[0] == self.N
            x[0, :] = x0

        for t in tqdm(range(k)):
            x[t+1, :] = x[t, :] - self.laplacian(x[t, :])

        # If we only diffused once, remove the time dimension
        if k == 1:
            x = np.squeeze(x)

        return x

    # Returns the clique expansion graph of the hypergraph
    def clique_expansion(self):
        G = nx.Graph()
        G.add_nodes_from(np.arange(self.N))

        # Go through hyperedges, and add edges between nodes
        # within the same hyperedge.
        for hedge in self.hyperedges:
            for i in range(len(hedge)):
                for j in range(i+1, len(hedge)):
                    if not hedge[i] == hedge[j]:
                        if G.has_edge(hedge[i], hedge[j]):
                            G.edges[hedge[i], hedge[j]]['weight'] += 1
                        else:
                            G.add_edge(hedge[i], hedge[j], weight=1)
        return G

    # Returns the Laplacian of the clique expansion of the hypergraph
    def clique_laplacian(self, as_tensor=False):
        L_c = nx.normalized_laplacian_matrix(self.clique_expansion())
        # Return a sparse tensor
        if as_tensor:
            indices = np.nonzero(L_c)
            values = torch.tensor(L_c[indices]).squeeze()
            return torch.sparse_coo_tensor(np.array(indices), values, (L_c.shape[0], L_c.shape[1]))
        else:
            return L_c

    # Returns the line expansion graph of the hypergraph
    def line_expansion(self):
        G = nx.Graph()
        G.add_nodes_from(np.arange(self.M))

        # Go through hyperedges, and add edges between them if they
        # share nodes, with weight proportional to the size of the
        # intersection.
        for i in range(self.M):
            for j in range(i+1, self.M):
                G.add_edge(i, j, weight=len(set(self.hyperedges[i]).intersection(self.hyperedges[j])))
        return G

    # Returns the Laplacian of the line expansion of the hypergraph
    def line_laplacian(self, as_tensor=False):
        L_l = nx.normalized_laplacian_matrix(self.line_expansion())
        # Return a sparse tensor
        if as_tensor:
            indices = np.nonzero(L_l)
            values = torch.tensor(L_l[indices]).squeeze()
            return torch.sparse_coo_tensor(np.array(indices), values, (L_l.shape[0], L_l.shape[1]))
        else:
            return L_l

    # Returns the incidence matrix of the hypergraph
    def incidence_matrix(self, as_tensor=False):
        B = np.zeros((self.N, self.M))
        for i in range(self.M):
            B[self.hyperedges[i], i] = 1
        # Return a sparse tensor
        if as_tensor:
            indices = np.nonzero(B)
            values = torch.tensor(B[indices]).squeeze()
            return torch.sparse_coo_tensor(np.array(indices), values, (B.shape[0], B.shape[1]))
        else:
            return B

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
