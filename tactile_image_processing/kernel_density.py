"""
Author: John Lloyd
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd

import cv2

from vsp.video_stream import CvVideoCamera, CvVideoDisplay
from vsp.detector import CvBlobDetector
from vsp.encoder import KeypointEncoder
from vsp.view import KeypointView


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

    blob_detector_params = {
          'min_threshold': 118,
          'max_threshold': 188,
          'filter_by_color': True,
          'blob_color': 255,
          'filter_by_area': True,
          'min_area': 41,
          'max_area': 134.5,
          'filter_by_circularity': True,
          'min_circularity': 0.20,
          'filter_by_inertia': True,
          'min_inertia_ratio': 0.36,
          'filter_by_convexity': True,
          'min_convexity': 0.39,
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
        detector = CvBlobDetector(**blob_detector_params)
        encoder = KeypointEncoder()
        view = KeypointView(color=(0, 255, 0))
        display = CvVideoDisplay(name='preview')
        display.open()

        # initialise taxels
        x0, y0, x1, y1 = bbox
        x = np.linspace(x0, x1, taxel_array_length)
        y = np.linspace(y0, y1, taxel_array_length)
        X, Y = np.meshgrid(x, y)
        taxels = np.column_stack((X.flatten(), Y.flatten()))

        # get initial density
        frame = camera.read()
        keypoints = encoder.encode(detector.detect(frame))
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
            start_time = time.time()
            frame = camera.read()
            keypoints = encoder.encode(detector.detect(frame))
            frame = view.draw(frame, keypoints)
            display.write(frame)

            density = pin_density(keypoints[:, :2], taxels, kernel_width=kernel_width)
            density_delta = density - init_density

            img.set_data(density_delta.reshape((taxel_array_length, -1)))
            plt.draw()
            fig.canvas.flush_events()

            k = cv2.waitKey(10)
            if k == 27:  # Esc key to stop
                break

            print('FPS: ', 1.0 / (time.time() - start_time))

    finally:
        camera.close()
        display.close()


if __name__ == '__main__':
    main(
        camera_source=8,
        kernel_width=15,
        taxel_array_length=128,
        v_abs_max=5e-5,
        bbox=[80, 25, 530, 475],
    )
