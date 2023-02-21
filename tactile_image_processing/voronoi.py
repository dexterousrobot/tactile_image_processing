"""
Author: John Lloyd,  Nathan Lepora, Wen Fan, Anupam Gupta
"""
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy.spatial import Voronoi, ConvexHull, Delaunay

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcol
import matplotlib.cm as cm

import cv2

from vsp.video_stream import CvVideoCamera

from tactile_image_processing.image_transforms import process_image
from tactile_image_processing.pin_extraction_methods import BlobDetector
from tactile_image_processing.pin_extraction_methods import ContourBlobDetector
from tactile_image_processing.pin_extraction_methods import DoHDetector
from tactile_image_processing.pin_extraction_methods import PeakDetector
from tactile_image_processing.pin_extraction_methods import SkeletonizeDetector


def cal_3d_Voronoi(Axx_canon, Cxx_canon, Cyy_canon, pool_neighbours, num_interp_points):
    cen_coord_canon = np.zeros((len(Axx_canon), 2))

    for ii in range(len(Cxx_canon)):
        cen_coord_canon[ii, 0] = np.mean(Cxx_canon[ii])
        cen_coord_canon[ii, 1] = np.mean(Cyy_canon[ii])

    vorarea_canon = np.asarray(Axx_canon)

    # Interpolate voronoi cell areas over neighbours
    dist_neighbourM_canon = scipy.spatial.distance.cdist(cen_coord_canon, cen_coord_canon, metric='euclidean')

    vorarea_canon_interp = np.zeros(np.shape(vorarea_canon))  # interlolate

    for ii in range(len(dist_neighbourM_canon)):
        indx_neighbour_canon = np.argsort(dist_neighbourM_canon[ii, :])[1:pool_neighbours+1]  # index from close neighbor
        # vorarea_canon_interp[ii] = (vorarea_canon[ii]+ (np.divide(1, pool_neighbours)*np.sum(vorarea_canon[indx_neighbour_canon])))*np.divide(1, 2)
        vorarea_canon_interp[ii] = (1*vorarea_canon[ii] + np.sum(vorarea_canon[indx_neighbour_canon])
                                    )*np.divide(1, pool_neighbours+1)

    # Create uniform grid to interpolate on
    Xgrid, Ygrid = np.meshgrid(
        np.linspace(np.min(cen_coord_canon[:, 0]), np.max(cen_coord_canon[:, 0]), num_interp_points),
        np.linspace(np.min(cen_coord_canon[:, 1]), np.max(cen_coord_canon[:, 1]), num_interp_points))

    # fit surface
    Z_canon = scipy.interpolate.griddata(cen_coord_canon, vorarea_canon, (Xgrid, Ygrid),
                                         'cubic', fill_value=np.min(vorarea_canon))
    Z_canon = cv2.GaussianBlur(Z_canon, (5, 5), 0)
    Z_canon = Z_canon - np.min(Z_canon)
    Z_canon = Z_canon / np.max(Z_canon)

    return Xgrid, Ygrid, Z_canon


def plot_3d_Voronoi(Xgrid, Ygrid, Z_canon, vmin=0.1, vmax=0.6):

    figg = plt.figure(1, figsize=(8, 8))
    figg.subplots_adjust(wspace=0, hspace=0, top=0.99, bottom=0.01)

    ## Select one of the colormap types below
    # cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", ['g', 'b', 'r'])
    cmap = plt.cm.get_cmap('jet')
    # cmap = plt.cm.get_cmap('coolwarm')
    # cmap = plt.cm.get_cmap('seismic')
    # cmap = plt.cm.get_cmap('bwr')
    # cmap = plt.cm.get_cmap('hot')
    # cmap = plt.cm.get_cmap('Reds')
    # cmap = plt.cm.get_cmap('inferno')
    # cmap = plt.cm.get_cmap('magma')
    # cmap=plt.cm.get_cmap('rainbow')

    ax = figg.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(Xgrid, Ygrid, Z_canon, cmap=cmap,
                    linewidth=1, antialiased=True, shade=False, vmin=vmin, vmax=vmax)

    plt.axis('on')
    plt.tight_layout()
    # ax.view_init(elev=71, azim=-110)
    ax.view_init(elev=90, azim=-90)
    plt.show()


class PlotVoronoiGraph:
    def __init__(self, cell_areas, cellvert_xcoord, cellvert_ycoord, cell_center, nodes_edges):
        super(PlotVoronoiGraph, self).__init__()
        self.cell_areas = cell_areas
        self.vert_xcoord = cellvert_xcoord
        self.vert_ycoord = cellvert_ycoord
        self.cell_center = cell_center
        self.nodes_edges = nodes_edges

    # @staticmethod
    def plot_voronoi(self, cell_scale_fact=1):

        patches = []
        for cell_no in range(len(self.cell_areas)):
            poly_vertices_x = np.multiply(self.vert_xcoord[cell_no], cell_scale_fact)
            poly_vertices_y = np.multiply(self.vert_ycoord[cell_no], cell_scale_fact)
            poly_vertices = np.transpose(np.array([poly_vertices_x, poly_vertices_y]))
            patches.append(Polygon(poly_vertices, closed=True))

        p = PatchCollection(patches, alpha=0.55)
        # Make a user-defined colormap.
        cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ['r', 'b', "w"])
        cnorm = mcol.Normalize(vmin=0, vmax=1)
        cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
        color_scale_vec = [1]
        colors = cpick.to_rgba(color_scale_vec)
        p.set_color(colors)
        p.set_edgecolor([0, 0, 0])  # rgb value
        p.set_linewidth(2)
        figg, axx = plt.subplots(figsize=(8, 8))
        axx.add_collection(p)
        axx.autoscale()

        # draw nodes
        x_cs = []
        y_cs = []
        for i in range(len(self.cell_center)):
            x_c = self.cell_center[i][0]
            y_c = self.cell_center[i][1]
            x_cs.append(x_c)
            y_cs.append(y_c)

        plt.plot(x_cs, y_cs, 'o', markersize=5, color='b', zorder=1)

        # draw edges
        src_des_x = []
        src_des_y = []
        for i in range(len(self.nodes_edges)):
            src = self.nodes_edges[i][0]
            des = self.nodes_edges[i][1]
            src_x = self.cell_center[src][0]
            src_y = self.cell_center[src][1]
            des_x = self.cell_center[des][0]
            des_y = self.cell_center[des][1]
            src_des_x.append([src_x, des_x])
            src_des_y.append([src_y, des_y])

        for i in range(len(self.nodes_edges)):
            plt.plot(src_des_x[i], src_des_y[i], color='r', zorder=0)

        plt.axis('on')
        plt.axis('equal')
        plt.show()
        return p


class TransformVoronoi:

    def __init__(self, borderScale=1.1):
        super(TransformVoronoi, self).__init__()
        self.borderScale = borderScale
        self.medianX = []

    def transform(self, X):
        # Voronoi tesselation function
        # Input: X = (x,y) pin positions (centroids)
        # Returns: Y = (x,y, A) pin positions and cell areas
        #          C cells are lists of vertices

        X = np.squeeze(X)
        _, unindx = np.unique(X, return_index=True, axis=0)
        unindx = np.sort(unindx)
        X = X[unindx]
        X = (X - np.median(X))

        # apply voronoi to data + boundary: V vertices, C cells

        B = ConvexHull(X)  # calculate the convexhull to extract the outter circle nodes
        # tmp = B.vertices
        BX = np.transpose([X[B.vertices, 0] * self.borderScale, X[B.vertices, 1]
                          * self.borderScale])  # new border circle nodes
        X_BX = np.vstack((X, BX))  # combine the boundary nodes with the origin nodes

        vor = Voronoi(X_BX, qhull_options='Qbb')  # Voronoi vertices and Voronoi cell
        BV = vor.vertices  # Coordinates of the Voronoi vertices
        BC = vor.regions  # Indices of the Voronoi vertices forming each Voronoi region
        BXY = vor.points  # input points position, equal to X_BX
        BXY_index = vor.point_region  # index of region to each point, Index of the Voronoi region for each input point

        # prune edges outside boundary
        Cx = np.asarray([BV[BC[BXY_index[i]], 0] for i in range(len(X))], dtype=object)
        Cy = np.asarray([BV[BC[BXY_index[i]], 1] for i in range(len(X))], dtype=object)
        A = [TransformVoronoi.polyarea(Cx[indx], Cy[indx]) for indx, val in enumerate(Cx)]

        return A, Cx, Cy, X

    @staticmethod
    def polyarea(x, y):
        # computes area of polygons
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def remove_repeat_edge(raw_nodes_edges):

    c = np.array(raw_nodes_edges)
    x = c[:, 0] + c[:, 1]*1j
    idx = np.unique(x, return_index=True)[1]
    clean_nodes_edges = c[idx]

    return clean_nodes_edges


def delaunay_graph_generate(nodes_pos):

    tri = Delaunay(nodes_pos)
    node_edges = []
    for i in range(len(tri.simplices)):
        for triangle in itertools.product(tri.simplices[i], tri.simplices[i]):
            if triangle[0] != triangle[1]:
                node_edges.append(triangle)
    node_edges = remove_repeat_edge(node_edges)

    return node_edges


def voronoi_graph_built(nodes_pos):

    nodes_pos_norm = np.squeeze(nodes_pos)
    _, unindx = np.unique(nodes_pos_norm, return_index=True, axis=0)
    unindx = np.sort(unindx)
    nodes_pos_norm = nodes_pos_norm[unindx]
    nodes_pos_norm = (nodes_pos_norm - np.median(nodes_pos_norm))

    clean_nodes_edges = delaunay_graph_generate(nodes_pos_norm)

    return clean_nodes_edges, nodes_pos_norm


def construct_Voronoi_graph_from_nodes_pos(
    pin_positions,
    image_dir,
    kernel_size=2,
    resize_x=300,
    resize_y=300,
    tip_num=331,
    k=6
):

    # Calculate voronoi tesseleation
    Axx_canon, Cxx_canon, Cyy_canon, XY_canon = TransformVoronoi(borderScale=1.1).transform(pin_positions)

    clean_nodes_edges, nodes_pos_norm = voronoi_graph_built(pin_positions)

    return Axx_canon, Cxx_canon, Cyy_canon, XY_canon, clean_nodes_edges


def main(
    camera_source=8,
    kernel_width=15,
    taxel_array_length=128,
    v_abs_max=5e-5,
    bbox=[80, 25, 530, 475],
):
    image_processing_params = {
        'dims': (256, 256),
        'bbox': [75, 30, 525, 480],
        'thresh': [11, -30],
        'stdiz': False,
        'normlz': True,
        'circle_mask_radius': 180,
    }

    try:
        # Windows
        # camera = CvVideoCamera(source=camera_source, api_name='DSHOW', is_color=False)

        # Linux
        camera = CvVideoCamera(source=camera_source, frame_size=(640, 480), is_color=False)
        camera.set_property('PROP_BUFFERSIZE', 1)
        for j in range(10):
            camera.read()   # dump previous frame because using first frame as baseline

        # set keypoint tracker
        # detector = BlobDetector()
        # detector = ContourBlobDetector()
        # detector = DoHDetector()
        # detector = PeakDetector()
        detector = SkeletonizeDetector()

        while True:
            start_time = time.time()
            raw_image = camera.read()

            processed_image = process_image(
                raw_image.copy(),
                gray=False,
                **image_processing_params
            )

            # keypoints = detector.extract_keypoints(raw_image)
            keypoints = detector.extract_keypoints(processed_image)

            # extract voronoi tesselation
            Axx_canon, Cxx_canon, Cyy_canon, XY_canon = TransformVoronoi(borderScale=1.1).transform(keypoints[:, :2])
            print('Axx_canon shape', len(Axx_canon))
            print('XY_canon shape', len(XY_canon))

            clean_nodes_edges, nodes_pos_norm = voronoi_graph_built(keypoints[:, :2])
            PlotVoronoiGraph(
                cell_areas=Axx_canon,
                cellvert_xcoord=Cxx_canon,
                cellvert_ycoord=Cyy_canon,
                cell_center=XY_canon,
                nodes_edges=clean_nodes_edges
            ).plot_voronoi(cell_scale_fact=1)

            Xgrid, Ygrid, Z_canon = cal_3d_Voronoi(
                Axx_canon,
                Cxx_canon,
                Cyy_canon,
                pool_neighbours=3,
                num_interp_points=50
            )
            plot_3d_Voronoi(Xgrid, Ygrid, Z_canon, vmin=0.1, vmax=0.9)

            k = cv2.waitKey(10)
            if k == 27:  # Esc key to stop
                break

            print('FPS: ', 1.0 / (time.time() - start_time))

    finally:
        camera.close()
        detector.display.close()


if __name__ == '__main__':
    main(
        camera_source=8,
        kernel_width=15,
        taxel_array_length=128,
        v_abs_max=5e-5,
        bbox=[80, 25, 530, 475],
    )
