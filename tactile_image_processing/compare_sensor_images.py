import numpy as np
import cv2
import matplotlib.pyplot as plt

from vsp.video_stream import CvVideoCamera
from vsp.processor import CameraStreamProcessorMT, AsyncProcessor

from ft_learn.utils.image_transforms import process_image

from ft_learn.utils.minitip_sensor_parameters import BOUNDING_BOXES
from ft_learn.utils.minitip_sensor_parameters import CIRCLE_MASK_RADIUS
from ft_learn.utils.minitip_sensor_parameters import FRAME_SIZE
from ft_learn.utils.minitip_sensor_parameters import BRIGHTNESS
from ft_learn.utils.minitip_sensor_parameters import CONTRAST
from ft_learn.utils.minitip_sensor_parameters import EXPOSURE
from ft_learn.utils.minitip_sensor_parameters import IS_COLOR


def make_sensor(source=0):
    sensor = AsyncProcessor(
                CameraStreamProcessorMT(
                    camera=CvVideoCamera(
                        source=source,
                        frame_size=FRAME_SIZE,
                        brightness=BRIGHTNESS,
                        contrast=CONTRAST,
                        exposure=EXPOSURE,
                        is_color=IS_COLOR,
                    )
                ))
    sensor.camera.set_property('PROP_FOURCC', cv2.VideoWriter_fourcc(*'MJPG'))
    return sensor


resize_dim = [256, 256]
sensor_name = 'sensor_1'
# sensor_name = 'sensor_2'
# sensor_name = 'sensor_3'
thresh = False
bboxs = BOUNDING_BOXES
circle_mask_radius = CIRCLE_MASK_RADIUS
# circle_mask_radius = 12

video_dirs = [
    # r'C:\Users\ac14293\Documents\tactip_data\ft_learn_data\async_ft_surface\sensor_1\lin_shear\train\videos\video_1.mp4',
    # r'C:\Users\ac14293\Documents\tactip_data\ft_learn_data\async_ft_surface\sensor_1\lin_shear\val\videos\video_1.mp4',
    # r'C:\Users\ac14293\Documents\tactip_data\ft_learn_data\async_ft_surface\sensor_2\lin_shear\train\videos\video_1.mp4',
    # r'C:\Users\ac14293\Documents\tactip_data\ft_learn_data\async_ft_surface\sensor_2\lin_shear\val\videos\video_1.mp4',
    # r'C:\Users\ac14293\Documents\tactip_data\ft_learn_data\async_ft_surface\sensor_3\lin_shear\train\videos\video_1.mp4',
    # r'C:\Users\ac14293\Documents\tactip_data\ft_learn_data\async_ft_surface\sensor_3\lin_shear\val\videos\video_1.mp4',

    r'/home/alex/Documents/tactip_datasets/ft_learn_data/async_ft_surface/sensor_1/lin_shear/train/videos/video_1.mp4',
    # r'/home/alex/Documents/tactip_datasets/ft_learn_data/async_ft_surface/sensor_1/lin_shear/val/videos/video_1.mp4',
]

# load from file
comparison_image = np.zeros(shape=resize_dim)
for video_dir, bbox in zip(video_dirs, bboxs):
    cap = cv2.VideoCapture(video_dir)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, init_frame_raw = cap.read()

    # preprocess/augment image
    init_frame_processed = process_image(
        init_frame_raw,
        gray=True,
        bbox=bbox,
        dims=resize_dim,
        thresh=thresh,
        circle_mask_radius=circle_mask_radius,
    )

    comparison_image += (init_frame_processed.squeeze() / 255.0) * 0.33

# live capture
compare_live = True
if compare_live:
    bbox = [129, 53, 514, 453]
    sensor = make_sensor(source=8)

    raw_frames = sensor.process(num_frames=1)
    init_frame_raw = raw_frames[0]
    init_frame_processed = process_image(
        init_frame_raw,
        gray=True,
        bbox=bbox,
        dims=resize_dim,
        thresh=thresh,
        circle_mask_radius=circle_mask_radius,
    )

    comparison_image += (init_frame_processed.squeeze() / 255.0) * 0.33

plt.imshow((comparison_image * 255).astype(np.uint8))
plt.show()
