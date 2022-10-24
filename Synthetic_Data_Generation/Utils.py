import matplotlib
import numpy as np
import hypernetx as hnx
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PatchCollection
import os


# Compute the spectral similarity between two square, symmetric, positive semi-definite, irreducible
# matrices of the same size.
# Finds the largest value epsilon (eps) such that (1 - eps)*lambda_i(A) <= lambda_i(B) <= (1 + eps)*lambda_i(A)
def spectral_similarity(A, B):
    # Compute eigenvalues, ignoring the smallest (since it is zero, up to numerical precision)
    eigs_A = np.linalg.eigvalsh(A)[1:]
    eigs_B = np.linalg.eigvalsh(B)[1:]

    # For each pair in ascending order, compute the lower bound for epsilon
    eps_vals = (eigs_B - eigs_A) / eigs_A

    # Pick the greatest lower bound
    return np.max(eps_vals)


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
              'with_node_labels': False, 'with_edge_labels': False}
    if node_pos is not None:
        kwargs['pos'] = {i: node_pos[i][:2] for i in range(H.N)}
    else:
        kwargs['layout_kwargs'] = {'seed': 40}
    hnx.drawing.draw(H_draw.collapse_edges(), **kwargs)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'hypergraph_sources.png'), dpi=200)
    else:
        plt.show()


# Plot the hypergraph and associated diffusions
def plot_diffusions_hg(SC, H, sourceEdges, tMax=10, signals_dict=None, save_dir=None):
    figSize = 7  # Overall size of the figure that contains the plot
    fig = plt.figure(figsize=(3.22 * figSize, 2.4 * figSize))
    nRows = np.ceil(len(sourceEdges)/2).astype(int) + 1

    # Plot the simplicial complex
    if SC is not None:
        sub1 = fig.add_subplot(nRows, 2, 1, projection='3d')
        sub1.axis('off')
        points = SC.pts
        sub1.scatter(points[:,0], points[:,1], points[:,2], color='black')
        sub1.set_box_aspect([1, 1, 0.2])
        bds = sub1.get_position().bounds
        scale = 3
        sub1.set_position([bds[0] - bds[2] * (scale-1)/2, bds[1] - bds[3] * (scale-1)/2, bds[2] * scale, bds[3] * scale])

        for line in SC.n_faces(1):
            x1, y1, z1 = points[line[0],:]
            x2, y2, z2 = points[line[1],:]
            sub1.plot([x1, x2], [y1, y2], [z1, z2], color='#000080')

        triangles = []
        for tri in SC.n_faces(2):
            sub1.add_collection3d(Poly3DCollection(points[tri,:], cmap=matplotlib.cm.jet, alpha=0.4))
        sub1.set_title('Cech Complex')

    # Plot the hypergraph
    sub2 = fig.add_subplot(nRows, 2, 2)
    H_draw = hnx.Hypergraph(H.hyperedges)
    kwargs = {'edges_kwargs': {'facecolors': [[1, 0, 0, 0.4] if e in sourceEdges else [0, 0, 0, 0]
                                              for e in np.arange(H.M)]},
              'with_node_labels': False, 'with_edge_labels': False,
              'ax': sub2, 'layout_kwargs': {'seed': 40}}
    hnx.drawing.draw(H_draw.collapse_edges(), **kwargs)
    sub2.set_title('Hypergraph ({} nodes, {} hyperedges)'.format(H.N, H.M))

    plot_index = 3
    for hedge_ind in sourceEdges:
        # Check if we were given a step size and should make our own diffusions, or if they were provided
        if signals_dict is not None:
            X = signals_dict[hedge_ind]
            steps = steps = np.tile(np.arange(0, X.shape[0]), (H.N, 1)).T
        else:
            # Construct the initial node signal
            hedge = H.hyperedges[hedge_ind]
            x0 = np.array([1 if node_ind in hedge else 0 for node_ind in range(H.N)])

            # Diffuse the node signal for tMax steps
            X = H.diffuse(x0, tMax)
            X = np.concatenate((x0[np.newaxis, :], X))
            steps = np.tile(np.arange(0, tMax + 1), (H.N, 1)).T

        # Plot the diffused signals
        sub = fig.add_subplot(nRows, 2, plot_index)
        sub.plot(steps, X)
        sub.set_title('Hyperedge {}'.format(hedge_ind))
        sub.set_ylabel('Signal')
        plot_index += 1

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, 'diffusions.png'), dpi=200)
    else:
        plt.show()


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
