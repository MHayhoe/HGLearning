import matplotlib
import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os


# Plots the diffusion of the signal x on the hypergraph H, according to L_H
def plot_diffusion(H, x, num_steps=1):
    # Plot the hypergraph, and color nodes according to signal values
    H_draw = hnx.Hypergraph(H.hyperedges)
    kwargs = {'layout_kwargs': {'seed': 40}, 'with_node_counts': False}
    color_map = plt.cm.viridis
    normed = plt.Normalize(x.min(), x.max())
    plt.subplot(211)
    hnx.drawing.draw(H_draw, nodes_kwargs={'facecolors': color_map(normed(x)) * (1, 1, 1, 1)}, **kwargs)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=normed,cmap=color_map))
    plt.draw()

    # Perform the diffusion
    X = np.zeros((num_steps+1,x.shape[0]))
    X[0,:] = x

    for t in range(1,num_steps+1):
        X[t,:] = H.diffuse(X[t-1,:])

    # Plot the diffused signals
    steps = np.tile(np.arange(0,num_steps+1), (H.N, 1)).T
    plt.subplot(212)
    plt.plot(steps, X)
    plt.legend(np.arange(1,H.N+1),loc='upper left')
    plt.show()


# Plot a hypergraph in 2 dimensions
def plot_2d_hg(H, node_pos=None, markedHedges=[], save_dir=None):
    plt.clf()
    H_draw = hnx.Hypergraph(H.hyperedges)
    kwargs = {'edges_kwargs': {'facecolors': [[1, 0, 0, 0.4] if e in markedHedges else [0, 0, 0, 0]
                                              for e in np.arange(H.M)]},
              'with_node_counts': False}
    if node_pos is not None:
        kwargs['pos'] = {i: node_pos[i][:2] for i in range(H.N)}
    else:
        kwargs['layout_kwargs'] = {'seed': 40}
    hnx.drawing.draw(H_draw.collapse_edges(), **kwargs)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'hypergraph_sources.png'), dpi=200)
    else:
        plt.show()
    plt.close()


# Plot a simplicial complex in 2 dimensions
def plot_2d_sc(sc, save_dir=None):
    plt.clf()
    fig, ax = plt.subplots()
    points = sc.pts[:,:2]
    ax.scatter(points[:,0], points[:,1], color='black')

    for line in sc.n_faces(1):
        x1, y1 = points[line[0],:]
        x2, y2 = points[line[1],:]
        ax.plot([x1,x2],[y1,y2],color = '#000080')

    triangles = []
    for tri in sc.n_faces(2):
        polygon = Polygon(points[tri,:], True, color='#89CFF0')
        triangles.append(polygon)

    p = PatchCollection(triangles, cmap=matplotlib.cm.jet, alpha=0.4)
    ax.add_collection(p)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'cech_complex.png'), dpi=200)
    else:
        plt.show(fig)
    plt.close(fig)
