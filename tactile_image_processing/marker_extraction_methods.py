import cv2

from vsp.video_stream import CvVideoCamera, CvVideoDisplay
from vsp.detector import CvBlobDetector
from vsp.detector import CvContourBlobDetector
from vsp.detector import SklDoHBlobDetector
from vsp.detector import SkeletonizePeakDetector
from vsp.encoder import KeypointEncoder
from vsp.view import KeypointView


class BlobDetector():
    """
    Author: John Lloyd
    """

    def __init__(self):
        self.blob_detector_params = {
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
              'blur_kernel_size': 7,
              'thresh_block_size': 15,
              'thresh_constant': -16.0,
              'min_radius': 4,
              'max_radius': 7,
          }

        # set keypoint tracker
        self.detector = CvContourBlobDetector(**self.blob_detector_params)
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

    def __init__(self):
        self.peak_detector_params = {
              'blur_kernel_size': 9,
              'min_distance': 10,
              'threshold_abs': 0.4346,
              'num_peaks': 331,
              'thresh_block_size': 11,
              'thresh_constant': -34.0,
        }

        # set keypoint tracker
        self.detector = SkeletonizePeakDetector(**self.peak_detector_params)
        self.encoder = KeypointEncoder()
        self.view = KeypointView(color=(0, 0, 255))
        self.display = CvVideoDisplay(name='skeletonize_peak')
        self.display.open()

    def extract_keypoints(self, image):
        keypoints = self.encoder.encode(self.detector.detect(image))
        image = self.view.draw(image, keypoints)
        self.display.write(image)
        return keypoints


def main(camera_source=8):

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

        while True:
            frame = camera.read()

            # apply keypoint extraction methods
            blob_detector.extract_keypoints(frame)
            contour_blob_detector.extract_keypoints(frame)
            doh_detector.extract_keypoints(frame)
            peak_detector.extract_keypoints(frame)

            k = cv2.waitKey(10)
            if k == 27:  # Esc key to stop
                break

    finally:
        camera.close()
        blob_detector.display.close()
        contour_blob_detector.display.close()
        doh_detector.display.close()
        peak_detector.display.close()


if __name__ == '__main__':
    main(camera_source=8)
