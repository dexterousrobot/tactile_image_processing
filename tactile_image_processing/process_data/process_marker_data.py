import os
import warnings
import cv2
import numpy as np
import pandas as pd

from tactile_data.utils import save_json_obj, load_json_obj
from tactile_data.collect_data.process_image_data import combine_bbox
from tactile_image_processing.image_transforms import process_image

from vsp.video_stream import CvVideoDisplay
from vsp.detector import CvBlobDetector
from vsp.detector import CvContourBlobDetector
from vsp.detector import SklDoHBlobDetector
from vsp.detector import SkeletonizePeakDetector
from vsp.encoder import KeypointEncoder
from vsp.view import KeypointView

warnings.simplefilter('always', UserWarning)

BASE_DATA_PATH = 'temp'


def process_marker_data(path, dir_names, marker_params={}, image_params={}):

    if type(dir_names) is str:
        dir_names = [dir_names]

    # set keypoint detector
    if marker_params['detector_type'] == 'blob':
        detector = CvBlobDetector(**marker_params['detector_kwargs'])
    elif marker_params['detector_type'] == 'contour':
        detector = CvContourBlobDetector(**marker_params['detector_kwargs'])
    elif marker_params['detector_type'] == 'doh':
        detector = SklDoHBlobDetector(**marker_params['detector_kwargs'])
    elif kp_process_params['detector_type'] == 'peak':
        detector = SkeletonizePeakDetector(**kp_process_params['detector_kwargs'])

    encoder = KeypointEncoder()
    view = KeypointView(color=(0, 255, 0))
    display = CvVideoDisplay(name='blob')
    display.open()

    # marker processing function 
    def process_kps(image, image_params={}):
        image = process_image(image, **image_params)
        if image.shape[2]==1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # extract keypoints; basic identification by sorting
        kps = encoder.encode(detector.detect(image))
        sorted_ind = np.lexsort((kps[:, 1], kps[:, 0]))
        kps = kps[sorted_ind]

        # display
        display.write(view.draw(image, kps))

        # normalise by image size and return first two components
        kps[:, 0] = kps[:, 0] / image.shape[1]
        kps[:, 1] = kps[:, 1] / image.shape[0]
        return kps[:, :2] 

    # iterate over dirs
    for dir_name in dir_names:

        # paths 
        image_dir = os.path.join(path, dir_name, 'sensor_images')
        kp_dir = os.path.join(path, dir_name, 'processed_markers')
        os.makedirs(kp_dir, exist_ok=True)

        # target files
        targets_df = pd.read_csv(os.path.join(path, dir_name, 'targets.csv'))     

        # process images to keypoints
        kp_filenames, kp_numbers = [], []
        for sensor_image in targets_df.sensor_image:
            image = cv2.imread(os.path.join(image_dir, sensor_image))
            kps = process_kps(image, image_params)

            # save as kp filename
            image_path, image_name = os.path.split(sensor_image)
            kp_filename = f"markers_{os.path.splitext(image_name)[0].split('_')[1]}.npy"
            np.save(os.path.join(kp_dir, kp_filename), kps)

            # store for targets
            kp_filenames.append(kp_filename)
            kp_numbers.append(kps.shape[0])

            # report 
            print(f'processed {dir_name}: {sensor_image} as {kp_filename} # markers {kps.shape[0]}')

        # try to process any zeroth/init images
        for image_name in ['image_0.png', 'frame_init_0.png']:
            kp_filename = f"markers_{os.path.splitext(image_name)[0].split('_')[1]}.npy"
            if os.path.isfile(os.path.join(image_dir, image_path, image_name)):
                image = cv2.imread(os.path.join(image_dir, sensor_image))
                kps = process_kps(image, image_params)
                np.save(os.path.join(kp_dir, kp_filename), kps)
                print(f'processed {dir_name}: {image_name} marker number {kps.shape[0]}')
                
        # add keypoint names to targets as targets_markers
        targets_df = targets_df.drop('sensor_image', axis=1)        
        targets_df.insert(loc=0, column="num_markers", value=kp_numbers)
        targets_df.insert(loc=0, column="markers_file", value=kp_filenames)
        targets_df.to_csv(os.path.join(path, dir_name, 'targets_markers.csv'), index=False)

        # save merged marker_params and image_params
        sensor_image_params = load_json_obj(os.path.join(path, dir_name, 'sensor_image_params'))
        processed_marker_params = {**marker_params, **sensor_image_params, **image_params}
        processed_marker_params['bbox'] = combine_bbox(sensor_image_params, image_params)    
        save_json_obj(processed_marker_params, os.path.join(path, dir_name, 'processed_marker_params'))


if __name__ == "__main__":

    dir_names = ["data_1", "data_2"]

    kp_process_params = {
        'n_pins': 127,
        # 'n_pins': 331,

        'image_width': 280,
        'image_height': 280,

        # 'detector_type': 'blob',
        # 'detector_kwargs': {
        #       'min_threshold': 82,
        #       'max_threshold': 205,
        #       'filter_by_color': True,
        #       'blob_color': 255,
        #       'filter_by_area': True,
        #       'min_area': 35,
        #       'max_area': 109,
        #       'filter_by_circularity': True,
        #       'min_circularity': 0.60,
        #       'filter_by_inertia': True,
        #       'min_inertia_ratio': 0.25,
        #       'filter_by_convexity': True,
        #       'min_convexity': 0.47,
        #   }

        # 'detector_type': 'contour',
        # 'detector_kwargs': {
        #     'blur_kernel_size': 7,
        #     'thresh_block_size': 15,
        #     'thresh_constant': -16.0,
        #     'min_radius': 4,
        #     'max_radius': 7,
        # },

        'detector_type': 'doh',
        'detector_kwargs': {
            'min_sigma': 5.0,
            'max_sigma': 6.0,
            'num_sigma': 5,
            'threshold': 0.015,
        },

        # 'detector_type': 'peak',
        # 'detector_kwargs': {
        #     'blur_kernel_size': 9,
        #     'min_distance': 10,
        #     'threshold_abs': 0.4346,
        #     'num_peaks': 331,
        #     'thresh_block_size': 11,
        #     'thresh_constant': -34.0,
        # },
    }

    process_marker_data(BASE_DATA_PATH, dir_names, kp_process_params)
