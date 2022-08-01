import cv2
import time
import numpy as np

ESC_KEY = 27
NO_KEY = -1
N_PREVIOUS_TIMESTAMPS_FPS = 5


class VideoCapture:
    def __init__(self, fps=12, src=0, width=640, height=480, exit_key=ESC_KEY):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.memory = None
        self.frame = None
        self.key_pressed = None
        self.fps = fps
        self.exit_key = exit_key

        # This is not accurate, but approximate TODO
        self.wait_key_delay = int((1 / fps) * 1000)

        self.previous_timestamps_fps = [None] * N_PREVIOUS_TIMESTAMPS_FPS

    def take_next(self):

        k = cv2.waitKey(self.wait_key_delay)
        if k == self.exit_key:
            return False  # TODO

        if not self.cap.isOpened():
            return False  # TODO

        success, frame = self.cap.read()
        if not success:
            return False  # TODO

        self.frame = frame
        self.key_pressed = k
        self.update_fps_timestamp()

        return True

    def update_fps_timestamp(self):
        self.previous_timestamps_fps.append(int(time.time() * 1000))
        self.previous_timestamps_fps.pop(0)

    def get_key(self):
        if self.key_pressed == NO_KEY:
            return None

        return chr(self.key_pressed)

    def get_fps(self):
        if not all(self.previous_timestamps_fps):
            return 0

        avg_ms = np.average(np.diff(self.previous_timestamps_fps))

        return round(1000 / avg_ms, 2)

    def get_frame(self):
        return self.frame

    def finish(self):
        self.frame = None
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        self.cap.release()

    def __exit__(self):
        self.cap.release()


def add_texts(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1

    y0, dy = 15, 15
    for i, line in enumerate(text):
        y = y0 + i * dy
        cv2.putText(
            img, line, (5, y), font, fontScale, color, thickness, cv2.LINE_AA
        )


def display_warning(img):
    img = cv2.circle(
        img,
        (int(img.shape[1] / 2), int(img.shape[0] / 2)),
        20,
        (0, 0, 255),
        -1,
    )

    return img
