"""
Author: John Lloyd
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd

import cv2

from vsp.video_stream import CvVideoCamera

from tactile_image_processing.marker_extraction_methods import BlobDetector
from tactile_image_processing.marker_extraction_methods import ContourBlobDetector
from tactile_image_processing.marker_extraction_methods import DoHDetector
from tactile_image_processing.marker_extraction_methods import PeakDetector


def pin_density(pin_positions, taxel_positions, kernel_width):
    # pin_positions = 331 x 2 array
    dist = ssd.cdist(taxel_positions, pin_positions, 'sqeuclidean')
    kernel = (1 / (2 * np.pi * kernel_width)) * np.exp(-dist / (2 * kernel_width ** 2))
    density = np.mean(kernel, axis=1)
    return density


def main(
    camera_source=8,
    kernel_width=15,
    taxel_array_length=128,
    v_abs_max=5e-5,
    bbox=[80, 25, 530, 475],
):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--detector',
        type=str,
        help="Choose device from ['blob', 'contour', 'doh', 'peak'].",
        default='blob'
    )
    args = parser.parse_args()

    # set keypoint detector
    if args.detector == 'blob':
        detector = BlobDetector()
    elif args.detector == 'contour':
        detector = ContourBlobDetector()
    elif args.detector == 'doh':
        detector = DoHDetector()
    elif args.detector == 'peak':
        detector = PeakDetector()

    try:
        # Windows
        # camera = CvVideoCamera(source=camera_source, api_name='DSHOW', is_color=False)

        # Linux
        camera = CvVideoCamera(source=camera_source, frame_size=(640, 480), is_color=False)
        camera.set_property('PROP_BUFFERSIZE', 1)
        for j in range(10):
            camera.read()   # dump previous frame because using first frame as baseline

        # initialise taxels
        x0, y0, x1, y1 = bbox
        x = np.linspace(x0, x1, taxel_array_length)
        y = np.linspace(y0, y1, taxel_array_length)
        X, Y = np.meshgrid(x, y)
        taxels = np.column_stack((X.flatten(), Y.flatten()))

        # get initial density
        frame = camera.read()
        keypoints = detector.extract_keypoints(frame)
        init_density = pin_density(keypoints[:, :2], taxels, kernel_width=kernel_width)

        # start a plot for density delta
        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(
            init_density.reshape((taxel_array_length, -1)),
            vmin=-v_abs_max,
            vmax=v_abs_max,
            cmap='jet'
        )
        ax.xaxis.set_ticks_position("top")
        plt.axis('scaled')
        plt.colorbar(img, ax=ax)
        plt.show(block=False)

        while True:
            frame = camera.read()
            keypoints = detector.extract_keypoints(frame)

            density = pin_density(keypoints[:, :2], taxels, kernel_width=kernel_width)
            density_delta = density - init_density

            img.set_data(density_delta.reshape((taxel_array_length, -1)))
            plt.draw()
            fig.canvas.flush_events()

            k = cv2.waitKey(10)
            if k == 27:  # Esc key to stop
                break

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
