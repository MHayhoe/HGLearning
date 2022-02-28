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

 
class CechComplex(SimplicialComplex):
  def __init__(self, points, epsilon, labels=None, distfcn=distance.euclidean):
    self.pts = points
    self.labels = list(range(len(self.pts))) if labels==None or len(labels)!=len(self.pts) else labels
    self.epsilon = epsilon
    self.distfcn = distfcn
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
    return g