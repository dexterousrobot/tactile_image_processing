import os
import shutil
import cv2
import numpy as np
import pandas as pd

from tactile_data.utils import save_json_obj, load_json_obj, make_dir
from tactile_image_processing.image_transforms import process_image

BASE_DATA_PATH = 'temp'


def process_image_data(path, dir_names, image_params={}):

    if type(dir_names) is str:
        dir_names = [dir_names]

    # iterate over dirs
    for dir_name in dir_names:

        # paths
        image_dir = os.path.join(path, dir_name, 'sensor_images')
        proc_image_dir = os.path.join(path, dir_name, 'processed_images')
        os.makedirs(proc_image_dir, exist_ok=True)

        # target files
        targets_df = pd.read_csv(os.path.join(path, dir_name, 'targets.csv'))

        # intitialise display
        cv2.namedWindow("processed_image")

        # process images
        image_filenames = []
        for sensor_image in targets_df.sensor_image:
            image = cv2.imread(os.path.join(image_dir, sensor_image))
            image = process_image(image, **image_params)
            
            # save as image filename
            image_path, image_name = os.path.split(sensor_image)
            image_filename = f"image_{os.path.splitext(image_name)[0].split('_')[1]}.png"
            cv2.imwrite(os.path.join(proc_image_dir, image_filename), image)

            # store for targets
            image_filenames.append(image_filename)

            # report
            print(f'processed {dir_name}: {image_name} as {image_filename}')
            cv2.imshow("processed_image", image)
            if cv2.waitKey(1)==27:    # Esc key to stop
                exit()

        # try to process any zeroth/init images
        for image_name in ['image_0.png', 'frame_init_0.png']:
            image_filename = f"image_{os.path.splitext(image_name)[0].split('_')[1]}.png"
            if os.path.isfile(os.path.join(image_dir, image_path, image_name)):
                image = cv2.imread(os.path.join(image_dir, sensor_image))
                image = process_image(image, **image_params)
                cv2.imwrite(os.path.join(proc_image_dir, image_filename), image)
                print(f'processed {dir_name}: {image_name} as {image_filename}')

        # new image names to targets_images
        targets_df["sensor_image"] = image_filenames
        targets_df.to_csv(os.path.join(path, dir_name, 'targets_images.csv'), index=False)

        # save merged sensor_params and image_params
        sensor_image_params = load_json_obj(os.path.join(path, dir_name, 'sensor_image_params'))
        proc_image_params = {**sensor_image_params, **image_params}
        proc_image_params['bbox'] = combine_bbox(sensor_image_params, image_params)
        save_json_obj(proc_image_params, os.path.join(path, dir_name, 'processed_image_params'))


def combine_bbox(image_params_1, image_params_2):
    b_1 = image_params_1.get('bbox', [0, 0, 0, 0])
    b_2 = image_params_2.get('bbox', [0, 0, 0, 0])
    return [b_1[0]+b_2[0], b_1[1]+b_2[1], b_1[0]+b_2[2], b_1[1]+b_2[3]]


def partition_data(path, dir_names, split=0.8, seed=1):

    if type(dir_names) is str:
        dir_names = [dir_names]

    if not split:
        return dir_names

    all_dirs_out = []
    for dir_name in dir_names:

        # load target df
        targets_df = pd.read_csv(os.path.join(path, dir_name, 'targets.csv'))

        # indices to split data
        np.random.seed(seed)  # make deternministic, needs to be different from collect
        inds_true = np.random.choice([True, False], size=len(targets_df), p=[split, 1-split])
        inds = [inds_true, ~inds_true]
        dirs_out = ['_'.join([out, dir_name]) for out in ["train", "val"]]

        # iterate over split
        for dir_out, ind in zip(dirs_out, inds):

            dir_out = os.path.join(path, dir_out)
            make_dir(dir_out, check=False)

            # copy over parameter files
            for filename in ['collect_params', 'env_params', 'sensor_image_params']:
                shutil.copy(os.path.join(path, dir_name, filename+'.json'), dir_out)

            # create dataframe pointing to original images (to avoid copying)
            rel_path = os.path.join('..', '..', dir_name, 'sensor_images', '') 
            targets_df.loc[ind, 'sensor_image'] = rel_path + targets_df[ind].sensor_image.map(str)
            targets_df[ind].to_csv(os.path.join(dir_out, 'targets.csv'), index=False)

        all_dirs_out = [*all_dirs_out, *dirs_out]

    return all_dirs_out


if __name__ == "__main__":

    dir_names = ["data_1", "data_2"]

    process_image_params = {
        'dims': (128, 128),
        "bbox": (12, 12, 240, 240)
    }

    # dir_names = split_data(BASE_DATA_PATH, dir_names)
    process_image_data(BASE_DATA_PATH, dir_names, process_image_params)
