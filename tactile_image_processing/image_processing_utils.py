import cv2
import numpy as np


class SimpleSensor:
    def __init__(
        self,
        source=0,
        exposure=-7,
    ):
        self.cam = cv2.VideoCapture(source)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, exposure)
        for _ in range(5):
            self.cam.read()  # Hack - camera transient

    def get_image(self):
        _, img = self.cam.read()
        return img


def list_camera_sources():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6:  # if there are more than 5 non working ports stop the testing.
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print(f"Port {dev_port} is not working.")
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print(f"Port {dev_port} is working and reads images ({h} x {w})")
                working_ports.append(dev_port)
            else:
                print(f"Port {dev_port} for camera ( {h} x {w}) is present but does not reads.")
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports, non_working_ports


def convert_image_uint8(image):
    image = (image-np.min(image))/(np.max(image)-np.min(image))
    image = 255 * image  # Now scale by 255
    return image.astype(np.uint8)


def pixel_diff_norm(frames):
    ''' Computes the mean pixel difference between the first frame and the
        remaining frames in a Numpy array of frames.
    '''
    n, h, w, c = frames.shape
    pdn = [cv2.norm(frames[i], frames[0], cv2.NORM_L1) / (h * w)
           for i in range(1, n)]
    return np.array(pdn)


def load_video_frames(filename):
    ''' Loads frames from specified video 'filename' and returns them as a
        Numpy array.
    '''
    frames = []
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        captured, frame = vc.read()
        if captured:
            frames.append(frame)
        while captured:
            captured, frame = vc.read()
            if captured:
                frames.append(frame)
        vc.release()
    return np.array(frames)


def camera_loop(
    camera,
    display_name='camera_display'
):
    cv2.namedWindow(display_name)
    while True:
        image = camera.get_image()
        cv2.imshow(display_name, image)
        k = cv2.waitKey(10)
        if k == 27:  # Esc key to stop
            break


if __name__ == '__main__':
    available_ports, working_ports, non_working_ports = list_camera_sources()

    print(f'Available Ports: {available_ports}')
    print(f'Working Ports: {working_ports}')
    print(f'Non-Working Ports: {non_working_ports}')

    source = 8
    if source not in working_ports:
        print(f'Camera port {source} not in working_ports: {working_ports}')

    camera = SimpleSensor(
        source=source
    )

    camera_loop(camera)
