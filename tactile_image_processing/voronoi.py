"""
Author: Luke Cramphorn, Nathan Lepora, Wen Fan, Anupam Gupta
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import Voronoi, ConvexHull, Delaunay
import cv2

from tactile_image_processing.marker_extraction_methods import MarkerDetector
from tactile_image_processing.simple_sensors import RealSensor


class MarkerVoronoi:
    def __init__(self):
        pass

    def transform(self, X, border_scale=1.1):
        """Voronoi tesselation function.

        Input: X = (x,y) pin positions (centroids)
        Returns: Y = (x,y, A) pin positions and cell areas
                 C cells are lists of vertices
        """
        X = np.squeeze(X[:, :2])
        _, unindx = np.unique(X, return_index=True, axis=0)
        unindx = np.sort(unindx)
        X = X[unindx]
        X = (X - np.median(X))

        # apply voronoi to data + boundary: V vertices, C cells
        B = ConvexHull(X)  # calculate the convexhull to extract the outter circle nodes

        # new border circle nodes
        BX = np.transpose([
            X[B.vertices, 0] * border_scale, X[B.vertices, 1] * border_scale
        ])
        X_BX = np.vstack((X, BX))  # combine the boundary nodes with the origin nodes

        vor = Voronoi(X_BX, qhull_options='Qbb')  # Voronoi vertices and Voronoi cell
        BV = vor.vertices  # Coordinates of the Voronoi vertices
        BC = vor.regions  # Indices of the Voronoi vertices forming each Voronoi region
        BXY_index = vor.point_region  # index of region to each point, Index of the Voronoi region for each input point

        # prune edges outside boundary
        Cx = np.asarray([BV[BC[BXY_index[i]], 0] for i in range(len(X))], dtype=object)
        Cy = np.asarray([BV[BC[BXY_index[i]], 1] for i in range(len(X))], dtype=object)
        A = [self.polyarea(Cx[indx], Cy[indx]) for indx, val in enumerate(Cx)]

        return A, Cx, Cy, X

    def polyarea(self, x, y):
        # computes area of polygons
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def remove_repeat_edge(self, raw_nodes_edges):

        c = np.array(raw_nodes_edges)
        x = c[:, 0] + c[:, 1]*1j
        idx = np.unique(x, return_index=True)[1]
        clean_nodes_edges = c[idx]

        return clean_nodes_edges

    def delaunay_graph_generate(self, nodes_pos):

        tri = Delaunay(nodes_pos)
        node_edges = []
        for i in range(len(tri.simplices)):
            for triangle in itertools.product(tri.simplices[i], tri.simplices[i]):
                if triangle[0] != triangle[1]:
                    node_edges.append(triangle)
        node_edges = self.remove_repeat_edge(node_edges)

        return node_edges

    def create_graph(self, nodes_pos):

        nodes_pos_norm = np.squeeze(nodes_pos)
        _, unindx = np.unique(nodes_pos_norm, return_index=True, axis=0)
        unindx = np.sort(unindx)
        nodes_pos_norm = nodes_pos_norm[unindx]
        nodes_pos_norm = (nodes_pos_norm - np.median(nodes_pos_norm))

        clean_nodes_edges = self.delaunay_graph_generate(nodes_pos_norm)

        return clean_nodes_edges, nodes_pos_norm

    def create_surface(self, Axx_canon, Cxx_canon, Cyy_canon, pool_neighbours, num_interp_points):
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
            vorarea_canon_interp[ii] = (1*vorarea_canon[ii] + np.sum(vorarea_canon[indx_neighbour_canon])
                                        )*np.divide(1, pool_neighbours+1)

        # Create uniform grid to interpolate on
        Xgrid, Ygrid = np.meshgrid(
            np.linspace(np.min(cen_coord_canon[:, 0]), np.max(cen_coord_canon[:, 0]), num_interp_points),
            np.linspace(np.min(cen_coord_canon[:, 1]), np.max(cen_coord_canon[:, 1]), num_interp_points)
        )

        # fit surface
        Z_canon = scipy.interpolate.griddata(cen_coord_canon, vorarea_canon, (Xgrid, Ygrid),
                                             'cubic', fill_value=np.min(vorarea_canon))
        Z_canon = cv2.GaussianBlur(Z_canon, (5, 5), 0)
        Z_canon = Z_canon - np.min(Z_canon)
        Z_canon = Z_canon / np.max(Z_canon)

        return Xgrid, Ygrid, Z_canon


class VoronoiPlotter:

    def __init__(self):

        # initialise plots
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))

        # setup graph plot
        self.axs[0].xaxis.set_ticks_position("top")
        self.axs[0].axis('on')
        self.axs[0].axis('equal')
        self.axs[0].invert_yaxis()

        # setup image plot
        self.axs[1].xaxis.set_ticks_position("top")
        self.axs[1].axis('on')
        self.axs[1].axis('equal')

        plt.show(block=False)

    def init_image(self, Z_canon):
        self.img = self.axs[1].imshow(
            Z_canon,
            vmin=0.0,
            vmax=1.0,
            cmap='jet',
            origin='lower'
        )
        plt.axis('scaled')
        plt.colorbar(self.img, ax=self.axs[1])

    def update_image(self, Z_canon):
        self.img.set_data(Z_canon)

    def plot_image(self, Z_canon):
        self.init_image(Z_canon)
        plt.show()

    def init_graph(
        self,
        cell_center,
        nodes_edges,
    ):

        # draw nodes
        x_node_data = []
        y_node_data = []
        for i in range(len(cell_center)):
            x_c = cell_center[i][0]
            y_c = cell_center[i][1]
            x_node_data.append(x_c)
            y_node_data.append(y_c)

        self.axs[0].plot(x_node_data, y_node_data, 'o', markersize=5, color='b', zorder=1)

        # draw edges
        x_edge_data = []
        y_edge_data = []
        for i in range(len(nodes_edges)):
            src = nodes_edges[i][0]
            des = nodes_edges[i][1]
            src_x = cell_center[src][0]
            src_y = cell_center[src][1]
            des_x = cell_center[des][0]
            des_y = cell_center[des][1]
            x_edge_data.append([src_x, des_x])
            y_edge_data.append([src_y, des_y])

        for i in range(len(nodes_edges)):
            self.axs[0].plot(x_edge_data[i], y_edge_data[i], color='r', zorder=0)

    def update_graph(
        self,
        cell_center,
        nodes_edges,
    ):
        self.axs[0].clear()
        self.init_graph(
            cell_center,
            nodes_edges,
        )

    def plot_graph(
        self,
        cell_center,
        nodes_edges,
    ):
        self.init_graph(
            cell_center,
            nodes_edges,
        )
        plt.show()


def voronoi_loop(camera, 
    num_interp_points=128,
    pool_neighbours=3,
    detector_type='doh',
    detector_kwargs=None
):

    if detector_kwargs:
        detector = MarkerDetector[detector_type](detector_kwargs)
    else:
        detector = MarkerDetector[detector_type]()

    marker_voronoi = MarkerVoronoi()

    # get initial voronoi map
    image = camera.process()
    if image.shape[2]==1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    keypoints = detector.extract_keypoints(image)

    Axx_canon, Cxx_canon, Cyy_canon, XY_canon = marker_voronoi.transform(keypoints, border_scale=1.1)

    nodes_edges, _ = marker_voronoi.create_graph(keypoints[:, :2])

    _, _, Z_canon = marker_voronoi.create_surface(
        Axx_canon,
        Cxx_canon,
        Cyy_canon,
        pool_neighbours=pool_neighbours,
        num_interp_points=num_interp_points
    )

    voronoi_plotter = VoronoiPlotter()
    voronoi_plotter.init_image(Z_canon)
    voronoi_plotter.init_graph(
        cell_center=XY_canon,
        nodes_edges=nodes_edges,
    )

    # start live camera loop
    try:
        while True:
            image = camera.process()
            if image.shape[2]==1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            keypoints = detector.extract_keypoints(image)

            # extract voronoi tesselation
            Axx_canon, Cxx_canon, Cyy_canon, XY_canon = marker_voronoi.transform(keypoints, border_scale=1.1)

            # create a graph from voronoi tesselation
            nodes_edges, _ = marker_voronoi.create_graph(keypoints[:, :2])

            # create a surface from voronoi tesselation
            _, _, Z_canon = marker_voronoi.create_surface(
                Axx_canon,
                Cxx_canon,
                Cyy_canon,
                pool_neighbours=pool_neighbours,
                num_interp_points=num_interp_points
            )

            # can be used to update the graph plot but this is slow
            voronoi_plotter.update_graph(
                cell_center=XY_canon,
                nodes_edges=nodes_edges,
            )

            voronoi_plotter.update_image(Z_canon)
            plt.draw()
            voronoi_plotter.fig.canvas.flush_events()

            if cv2.waitKey(10)==27:  # Esc key to stop
                break

    finally:
        detector.display.close()


if __name__ == '__main__':

    sensor_params = {
        "source": 1,
        "bbox": (160-15, 80+5, 480-15, 400+5),
        "circle_mask_radius": 155,
        "thresh": (11, -30)
    }

    camera = RealSensor(sensor_params)

    marker_kwargs = {
        'detector_type': 'doh',
        'detector_kwargs': {
            'min_sigma': 5,
            'max_sigma': 6,
            'num_sigma': 5,
            'threshold': 0.015,
        }
    }

    voronoi_kwargs = {
        'num_interp_points': 128,
        'pool_neighbours': 3
    }

    voronoi_loop(camera, **voronoi_kwargs, **marker_kwargs)
