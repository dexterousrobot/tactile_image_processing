import os
import pathlib
import pandas as pd
import cv2

BASE_DATA_PATH = 'temp'


def check_images_exist(path, dir_names):

    if type(dir_names) is str:
        dir_names = [dir_names]

    for dir_name in dir_names:

        # load target df
        targets_df = pd.read_csv(os.path.join(path, dir_name, 'targets.csv'))

        images_not_found = []

        for row in targets_df.iterrows():
            img_name = row[1]['sensor_image']
            print(f'checking {dir_name}: {img_name}')

            infile = os.path.join(path, dir_name, 'images', img_name)
            image = cv2.imread(infile)

            if image is None:
                images_not_found.append(img_name)
                print('Image not found')

        if images_not_found == []:
            print('Dataset is complete')
        else:
            print('Images not found: ', images_not_found)


def adjust_csv(path, dir_names, dry_run=True):

    if type(dir_names) is str:
        dir_names = [dir_names]

    def adjust_filename(video_filename):
        video_filename = pathlib.Path(video_filename).stem
        id = video_filename.split('_')[1]
        return f'image_{id}.png'

    for dir_name in dir_names:
        target_df = pd.read_csv(os.path.join(path, dir_name, 'targets_video.csv'))
        target_df['sensor_image'] = target_df.sensor_video.apply(adjust_filename)
        target_df.drop('sensor_video', axis=1, inplace=True)
        print(target_df)

        if not dry_run:
            target_df.to_csv(os.path.join(path, dir_name, 'targets.csv'), index=False)


if __name__ == '__main__':

    dir_names = ["data_1", "data_2"]
    dry_run = True

    check_images_exist(BASE_DATA_PATH, dir_names)
    # adjust_csv(BASE_DATA_PATH, dir_names, dry_run)
