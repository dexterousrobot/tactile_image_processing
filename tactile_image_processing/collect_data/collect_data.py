import os
import numpy as np

from tactile_data.collect_data.setup_embodiment import setup_embodiment
from tactile_data.collect_data.setup_targets import setup_targets
from tactile_data.collect_data.setup_targets import POSE_LABEL_NAMES, SHEAR_LABEL_NAMES, OBJECT_POSE_LABEL_NAMES
from tactile_data.utils import make_dir, save_json_obj

BASE_DATA_PATH = 'temp'


def collect_data(
    robot,
    sensor,
    targets_df,
    image_dir,
    collect_params,
):
    pose_label_names = collect_params.get('pose_label_names', POSE_LABEL_NAMES)
    shear_label_names = collect_params.get('shear_label_names', SHEAR_LABEL_NAMES)
    object_pose_label_names = collect_params.get('object_pose_label_names', OBJECT_POSE_LABEL_NAMES)

    # start 50mm above workframe origin with zero joint 6
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints([*robot.joint_angles[:-1], 0])

    # collect reference image
    image_outfile = os.path.join(image_dir, 'image_0.png')
    sensor.process(image_outfile)

    # clear object by 10mm
    clearance = (0, 0, 10, 0, 0, 0)
    robot.move_linear(np.zeros(6) - clearance)
    joint_angles = robot.joint_angles
    saved_obj_label = ''

    # ==== data collection loop ====
    for i, row in targets_df.iterrows():
        image_name = row.loc["sensor_image"]
        obj_label = row.loc["object_label"]
        pose = row.loc[pose_label_names].values.astype(float)
        shear = row.loc[shear_label_names].values.astype(float)
        obj_pose = row.loc[object_pose_label_names].values.astype(float)

        # report
        with np.printoptions(precision=1, suppress=True):
            print(f"{i+1}/{len(targets_df.index)}: [{obj_label}] pose{pose}, shear{shear}")

        # new object set reset
        if obj_label != saved_obj_label:
            saved_obj_label = obj_label
            robot.move_joints(joint_angles)
            robot.move_linear(obj_pose - clearance)
            joint_angles = robot.joint_angles

        # pose is relative to object pose
        pose += obj_pose

        # move to above new pose (avoid changing pose in contact with object)
        robot.move_linear(pose + shear - clearance)

        # move down to offset pose
        robot.move_linear(pose + shear)

        # move to target pose inducing shear
        robot.move_linear(pose)

        # collect and process tactile image
        image_outfile = os.path.join(image_dir, image_name)
        sensor.process(image_outfile)

        # move above the target pose
        robot.move_linear(pose - clearance)

        # if sorted, don't move to reset position
        if not collect_params.get('sort', False):
            robot.move_joints(joint_angles)

    # finish 50mm above workframe origin then zero last joint
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()


if __name__ == "__main__":

    data_params = {
        'data_1': 50,
        'data_2': 50,
    }

    collect_params = {
        "pose_llims": (-5, 0, 3, 0, 0, -180),
        "pose_ulims": (5, 0, 4, 0, 0,  180),
        "sort": True,
        "object_poses": {
            "edge":    (0, 0, 0, 0, 0, 0),
            "surface": (-50, 0, 0, 0, 0, 0)
        }
    }

    env_params = {
        "robot": "sim",
        "stim_name": "square",
        "work_frame": (650, 0, 50, -180, 0, 0),
        "tcp_pose":   (0, 0, -85, 0, 0, 0),
        "stim_pose":  (600, 0, 12.5, 0, 0, 0),
        'show_tactile': True
    }

    sensor_params = {
        "type": "standard_tactip",
        "image_size": (256, 256)
    }

    for data_dir_name, num_poses in data_params.items():

        # setup save dir
        save_dir = os.path.join(BASE_DATA_PATH, data_dir_name)
        image_dir = os.path.join(save_dir, "sensor_images")
        make_dir(save_dir)
        make_dir(image_dir)
        save_json_obj(sensor_params, os.path.join(save_dir, 'sensor_params'))

        # setup embodiment
        robot, sensor = setup_embodiment(
            env_params,
            sensor_params
        )

        # setup targets to collect
        target_df = setup_targets(
            collect_params,
            num_poses,
            save_dir
        )

        # collect
        collect_data(
            robot,
            sensor,
            target_df,
            image_dir,
            collect_params
        )
