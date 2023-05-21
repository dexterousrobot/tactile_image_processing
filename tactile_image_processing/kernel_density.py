"""
Author: John Lloyd
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import cv2

from tactile_image_processing.marker_extraction_methods import MarkerDetector
from tactile_image_processing.simple_sensors import RealSensor


class MarkerDensity():

    def __init__(self,
        grid_size=(100,100),
        kernel_width=15,
        normalization=5e-5,
        bbox=(640,480),
    ):
        x = np.linspace(0, bbox[2]-bbox[0], grid_size[0])
        y = np.linspace(0, bbox[3]-bbox[1], grid_size[1])
        X, Y = np.meshgrid(x, y)

        self.grid = np.column_stack((X.flatten(), Y.flatten()))
        self.grid_size = grid_size
        self.norm = normalization
        self.kernel_width = kernel_width

    def extract(self, keypoints):
        dist = ssd.cdist(self.grid, keypoints[:, :2], 'sqeuclidean')    # keypoints = n_markers x 2 array
        kernel = (1/(2*np.pi*self.kernel_width)) * np.exp(-dist/(2*self.kernel_width**2))
        density = np.mean(kernel, axis=1)/self.norm
        return density.reshape((self.grid_size[1], self.grid_size[0]))


def apply_circle_mask(image, radius=100):
    hh, ww = image.shape[:2]
    mask = np.ones(shape=(hh, ww))
    mask = cv2.circle(mask, (ww//2, hh//2), radius, 0, -1)
    mask[mask==1] = np.nan
    mask[mask==0] = 1
    return image * mask


def camera_loop(camera, density_kwargs, 
        detector_type='doh',
        detector_kwargs=None
    ):
    
    if detector_kwargs:
        detector = MarkerDetector[detector_type](detector_kwargs)
    else:
        detector = MarkerDetector[detector_type]()

    # initialise marker density
    marker_density = MarkerDensity(**density_kwargs)

    # get initial density
    image = camera.process()
    if image.shape[2]==1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    keypoints = detector.extract_keypoints(image)
    init_density = marker_density.extract(keypoints)
    init_density = apply_circle_mask(init_density, 100)

    # start a plot for density delta
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    img = ax.imshow(init_density, vmin=-1, vmax=1, cmap='jet')
    plt.axis('off')
    plt.axis('scaled')
    plt.show(block=False)

    try:
        while True:
            image = camera.process()
            if image.shape[2]==1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            keypoints = detector.extract_keypoints(image)
            density = marker_density.extract(keypoints)
            density = apply_circle_mask(density, 100)

            img.set_data(density - init_density)
            plt.draw()
            fig.canvas.flush_events()

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

    density_params = {
        'kernel_width': 15,
        'grid_size': (200, 200),
        'normalization':5e-5,
        'bbox': sensor_params['bbox']
    }

    camera_loop(camera, density_params, **marker_kwargs)
