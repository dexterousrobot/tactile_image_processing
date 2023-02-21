import time
import numpy as np
from skimage.feature import corner_peaks, peak_local_max
from skimage.morphology import medial_axis
import cv2

from tactile_image_processing.image_transforms import process_image

from vsp.video_stream import CvVideoCamera, CvVideoDisplay
from vsp.detector import CvBlobDetector
from vsp.detector import CvContourBlobDetector
from vsp.detector import SklDoHBlobDetector
from vsp.encoder import KeypointEncoder
from vsp.view import KeypointView
from vsp.feature import Keypoint


class BlobDetector():
    """
    Author: John Lloyd
    """

    def __init__(self):
        self.blob_detector_params = {
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

        # set keypoint tracker
        self.detector = CvBlobDetector(**self.blob_detector_params)
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

    def __init__(self):
        self.blob_detector_params = {
              'blur_kernel_size': 9,
              'min_threshold': 44,
              'max_threshold': 187,
              'min_radius': 2,
              'max_radius': 20,
          }

        # set keypoint tracker
        self.detector = CvContourBlobDetector(**self.blob_detector_params)
        self.encoder = KeypointEncoder()
        self.view = KeypointView(color=(255, 0, 0))
        self.display = CvVideoDisplay(name='contour_blob')
        self.display.open()

    def extract_keypoints(self, image):
        image = np.squeeze(np.uint8(image*255))
        keypoints = self.encoder.encode(self.detector.detect(image))
        image = self.view.draw(image, keypoints)
        self.display.write(image)
        return keypoints


class DoHDetector():
    """
    Author: John Lloyd
    """

    def __init__(self):
        self.doh_detector_params = {
              'min_sigma': 5.0,
              'max_sigma': 6.0,
              'num_sigma': 5,
              'threshold': 0.015,
          }

        # set keypoint tracker
        self.detector = SklDoHBlobDetector(**self.doh_detector_params)
        self.encoder = KeypointEncoder()
        self.view = KeypointView(color=(0, 255, 255))
        self.display = CvVideoDisplay(name='DoH')
        self.display.open()

    def extract_keypoints(self, image):
        image = np.squeeze(np.uint8(image*255))
        keypoints = self.encoder.encode(self.detector.detect(image))
        image = self.view.draw(image, keypoints)
        self.display.write(image)
        return keypoints


class PeakDetector():
    """
    Author: Anupam Gupta
    """

    def __init__(self):
        self.peak_detector_params = {
              'blur_ker': 5,
              'min_distance': 6,
              'kp_size': 2.0,
        }

        # set keypoint tracker
        self.encoder = KeypointEncoder()
        self.view = KeypointView(color=(0, 0, 255))
        self.display = CvVideoDisplay(name='peak')
        self.display.open()

    def detect(self, image):
        img_blur = cv2.medianBlur(src=image, ksize=self.peak_detector_params['blur_ker'])
        img_blur = img_blur - np.min(img_blur)
        img_blur = img_blur / np.max(img_blur)
        keypoints = np.fliplr(corner_peaks(img_blur, min_distance=self.peak_detector_params['min_distance']))

        # add sizes to keypoints
        sizes = np.ones(keypoints.shape[0]) * self.peak_detector_params['kp_size']
        keypoints = np.hstack([keypoints, sizes[..., np.newaxis]])

        # convert to correct type for drawing
        keypoints = [cv2.KeyPoint(x[0], x[1], x[2]) for x in keypoints]
        keypoints = [Keypoint(kp.pt, kp.size) for kp in keypoints]

        return keypoints

    def extract_keypoints(self, image):
        image = np.squeeze(np.uint8(image*255))
        keypoints = self.encoder.encode(self.detect(image))
        image = self.view.draw(image, keypoints)
        self.display.write(image)
        return keypoints


class SkeletonizeDetector():
    """
    Author: Alex Church
    """

    def __init__(self):
        self.skeletonize_detector_params = {
            'min_distance': 6,
            'kp_size': 2.0,
        }

        # set keypoint tracker
        self.encoder = KeypointEncoder()
        self.view = KeypointView(color=(255, 255, 0))
        self.display = CvVideoDisplay(name='skeletonize')
        self.display.open()

    def detect(self, image):

        # Compute the medial axis (skeleton) and the distance transform
        image = np.squeeze(image)
        skel, distance = medial_axis(image, return_distance=True)
        image = distance * skel

        # detect keypoints as local peaks
        keypoints = np.fliplr(peak_local_max(image, min_distance=self.skeletonize_detector_params['min_distance']))

        # add sizes to keypoints
        sizes = np.ones(keypoints.shape[0]) * self.skeletonize_detector_params['kp_size']
        keypoints = np.hstack([keypoints, sizes[..., np.newaxis]])

        # convert to correct type for drawing
        keypoints = [cv2.KeyPoint(x[0], x[1], x[2]) for x in keypoints]
        keypoints = [Keypoint(kp.pt, kp.size) for kp in keypoints]

        return keypoints

    def extract_keypoints(self, image):
        image = np.uint8(image*255)
        keypoints = self.encoder.encode(self.detect(image))
        image = self.view.draw(image, keypoints)
        self.display.write(image)
        return keypoints


def main(
    camera_source=8,
    image_processing_params={},
):

    try:
        # Windows
        # camera = CvVideoCamera(source=camera_source, api_name='DSHOW', is_color=False)

        # Linux
        camera = CvVideoCamera(source=camera_source, frame_size=(640, 480), is_color=False)
        camera.set_property('PROP_BUFFERSIZE', 1)
        for j in range(10):
            camera.read()   # dump previous frame because using first frame as baseline

        # init blob detection
        blob_detector = BlobDetector()
        contour_blob_detector = ContourBlobDetector()
        doh_detector = DoHDetector()
        peak_detector = PeakDetector()
        skeletonize_detector = SkeletonizeDetector()

        while True:
            start_time = time.time()
            raw_image = camera.read()

            processed_image = process_image(
                raw_image.copy(),
                gray=False,
                **image_processing_params
            )

            # apply keypoint extraction methods
            blob_detector.extract_keypoints(raw_image)
            contour_blob_detector.extract_keypoints(processed_image)
            doh_detector.extract_keypoints(raw_image)
            peak_detector.extract_keypoints(processed_image)
            skeletonize_detector.extract_keypoints(processed_image)

            k = cv2.waitKey(10)
            if k == 27:  # Esc key to stop
                break

            print('FPS: ', 1.0 / (time.time() - start_time))

    finally:
        camera.close()
        blob_detector.display.close()
        contour_blob_detector.display.close()
        doh_detector.display.close()
        peak_detector.display.close()
        skeletonize_detector.display.close()


if __name__ == '__main__':
    image_processing_params = {
        'dims': (256, 256),
        'bbox': [75, 30, 525, 480],
        'thresh': [11, -30],
        'stdiz': False,
        'normlz': True,
        'circle_mask_radius': 180,
    }

    main(
        camera_source=8,
        image_processing_params=image_processing_params,
    )
