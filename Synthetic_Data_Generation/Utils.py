import matplotlib
import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


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


def plot_2d_sc(sc):
    fig, ax = plt.subplots()
    points = sc.pts
    plt.scatter(points[:,0], points[:,1])


    for line in sc.n_faces(1):
        x1, y1 = points[line[0],:]
        x2, y2 = points[line[1],:]
        plt.plot([x1,x2],[y1,y2],color = '#000080')

    triangles = []
    for tri in sc.n_faces(2):
        polygon = Polygon(points[tri,:], True, color='#89CFF0')
        triangles.append(polygon)

    p = PatchCollection(triangles, cmap=matplotlib.cm.jet, alpha=0.4)
    ax.add_collection(p)

    plt.show()