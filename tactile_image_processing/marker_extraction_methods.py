import cv2

from vsp.video_stream import CvVideoDisplay
from vsp.detector import CvBlobDetector
from vsp.detector import CvContourBlobDetector
from vsp.detector import SklDoHBlobDetector
from vsp.detector import SkeletonizePeakDetector
from vsp.encoder import KeypointEncoder
from vsp.view import KeypointView

from tactile_image_processing.simple_sensors import RealSensor


class BlobDetector():
    """
    Author: John Lloyd
    """

    def __init__(self,
        detector_kwargs = {
            'min_threshold': 82,
            'max_threshold': 205,
            'filter_by_color': True,
            'blob_color': 255,
            'filter_by_area': True,
            'min_area': 35,
            'max_area': 109,
            'filter_by_circularity': True,
            'min_circularity': 0.60,
            'filter_by_inertia': True,
            'min_inertia_ratio': 0.25,
            'filter_by_convexity': True,
            'min_convexity': 0.47,
        }             
    ):
        self.detector_kwargs = detector_kwargs
        self.detector = CvBlobDetector(**self.detector_kwargs)
        self.encoder = KeypointEncoder()
        self.view = KeypointView(color=(0, 255, 0))
        self.display = CvVideoDisplay(name='blob')
        self.display.open()

    def extract_keypoints(self, image):
        keypoints = self.encoder.encode(self.detector.detect(image))
        image = self.view.draw(image, keypoints)
        self.display.write(image)
        return keypoints


class ContourBlobDetector():
    """
    Author: John Lloyd
    """

    def __init__(self,
        detector_kwargs = {
            'blur_kernel_size': 7,
            'thresh_block_size': 15,
            'thresh_constant': -16.0,
            'min_radius': 4,
            'max_radius': 7,
        }                 
    ):
        self.detector_kwargs = detector_kwargs
        self.detector = CvContourBlobDetector(**self.detector_kwargs)
        self.encoder = KeypointEncoder()
        self.view = KeypointView(color=(255, 0, 0))
        self.display = CvVideoDisplay(name='contour_blob')
        self.display.open()

    def extract_keypoints(self, image):
        keypoints = self.encoder.encode(self.detector.detect(image))
        image = self.view.draw(image, keypoints)
        self.display.write(image)
        return keypoints


class DoHDetector():
    """
    Author: John Lloyd
    """

    def __init__(self,
        detector_kwargs = {
            'min_sigma': 5.0,
            'max_sigma': 6.0,
            'num_sigma': 5,
            'threshold': 0.015,
        }                 
    ):
        self.detector_kwargs = detector_kwargs
        self.detector = SklDoHBlobDetector(**self.detector_kwargs)
        self.encoder = KeypointEncoder()
        self.view = KeypointView(color=(255, 0, 255))
        self.display = CvVideoDisplay(name='DoH')
        self.display.open()

    def extract_keypoints(self, image):
        keypoints = self.encoder.encode(self.detector.detect(image))
        image = self.view.draw(image, keypoints)
        self.display.write(image)
        return keypoints


class PeakDetector():
    """
    Author: Anupam Gupta, Alex Church
    """

    def __init__(self, 
        detector_kwargs = {
            'blur_kernel_size': 9,
            'min_distance': 10,
            'threshold_abs': 0.4346,
            'num_peaks': 331,
            'thresh_block_size': 11,
            'thresh_constant': -34.0
        }
    ):
        self.detector_kwargs = detector_kwargs
        self.detector = SkeletonizePeakDetector(**self.detector_kwargs)
        self.encoder = KeypointEncoder()
        self.view = KeypointView(color=(0, 0, 255))
        self.display = CvVideoDisplay(name='skeletonize_peak')
        self.display.open()

    def extract_keypoints(self, image):
        keypoints = self.encoder.encode(self.detector.detect(image))
        image = self.view.draw(image, keypoints)
        self.display.write(image)
        return keypoints


MarkerDetector = {
    'blob': BlobDetector,
    'contour': ContourBlobDetector,
    'doh': DoHDetector,
    'peak': PeakDetector
}


def camera_loop(camera, 
        detector_type='doh',
        detector_kwargs=None
    ):

    if detector_kwargs:
        detector = MarkerDetector[detector_type](detector_kwargs)
    else:
        detector = MarkerDetector[detector_type]()

    try:
        while True:
            image = camera.process()
            if image.shape[2]==1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            detector.extract_keypoints(image)

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
            'min_sigma': 5.0,
            'max_sigma': 6.0,
            'num_sigma': 5,
            'threshold': 0.015,
        }
    }

    camera_loop(camera, **marker_kwargs)
