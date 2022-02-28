import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

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