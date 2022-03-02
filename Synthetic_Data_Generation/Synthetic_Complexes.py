import numpy as np
from itertools import combinations
from scipy.sparse import dok_matrix
from operator import add
import networkx as nx
from scipy.spatial import distance
from itertools import product
from tqdm import tqdm

## Code from https://datawarrior.wordpress.com/tag/cech-complex/


class SimplicialComplex:
  def __init__(self, simplices=[]):
    self.import_simplices(simplices=simplices)

  def import_simplices(self, simplices=[]):
    self.simplices = list(map(lambda simplex: tuple(sorted(simplex)), simplices))
    self.face_set = self.faces()
    self.bms = self.boundary_maps()
    self.hodge_laps = self.hodge_laplacians()

  def faces(self):
    faceset = set()
    for simplex in tqdm(self.simplices):
      numnodes = len(simplex)
      for r in range(numnodes, 0, -1):
        for face in combinations(simplex, r):
          faceset.add(face)
    return faceset

  def n_faces(self, n):
    return list(filter(lambda face: len(face)==n+1, self.face_set))

  def boundary_maps(self):
    # B_0 = 0
    maps = [0]
    order = 1
    while len(self.n_faces(order+1)) > 0:
      n_faces_O = self.n_faces(order)
      n_faces_OP1 = self.n_faces(order+1)
      bm = np.zeros((len(n_faces_O), len(n_faces_OP1)))

      for i, nfo in enumerate(n_faces_O):
        for j, nfo1 in enumerate(n_faces_OP1):
          if set(nfo).issubset(set(nfo1)): 
            # TO_DO - Figure out orientation problem
            bm[i,j] = 1
      maps.append(bm)

      order += 1

    return maps

  def hodge_laplacians(self):
    hodge_laps = []

    for order in range(len(self.bms) - 1):
      if order == len(self.bms):
        hl = self.bms[order].T @ self.bms[order]
      elif order == 0:
        hl = self.bms[order+1] @ self.bms[order+1].T
      else:
        hl = self.bms[order].T @ self.bms[order] + self.bms[order+1] @ self.bms[order+1].T
      hodge_laps.append(hl)

    return hodge_laps

 
class CechComplex(SimplicialComplex):
  def __init__(self, points, epsilon, labels=None, distfcn=distance.euclidean, lcc = False):
    self.pts = points
    self.labels = list(range(len(self.pts))) if labels==None or len(labels)!=len(self.pts) else labels
    self.epsilon = epsilon
    self.distfcn = distfcn
    self.lcc = lcc
    self.network = self.construct_network(self.pts, self.labels, self.epsilon, self.distfcn)
    self.import_simplices(map(tuple, list(nx.find_cliques(self.network))))
    
 
  def construct_network(self, points, labels, epsilon, distfcn):
    g = nx.Graph()
    g.add_nodes_from(labels)
    zips = list(zip(points, labels))
    
    for pair in tqdm(product(zips, zips)):
      if pair[0][1] != pair[1][1]:
        dist = distfcn(pair[0][0], pair[1][0])
        if dist <= epsilon:
          g.add_edge(pair[0][1], pair[1][1])
    
    if self.lcc:
      # Gets largest connected component
      gcc = sorted(nx.connected_components(g), key=len, reverse=True)
      g0 = g.subgraph(gcc[0])
      
      # Relabels nodes and edges
      nodelist = g0.nodes()
      self.pts = self.pts[g0.nodes(),:]
      nodelist.sort()
      mapping = {old_label:new_label for new_label, old_label in enumerate(nodelist)}
      g0rel = nx.relabel_nodes(g0, mapping)
      return g0rel
    return g