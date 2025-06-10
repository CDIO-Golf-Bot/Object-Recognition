# camera.py
import threading
import cv2
from queue import Queue

class Camera:
    def __init__(self, config):
        self.config = config
        self.frame_queue = Queue(maxsize=1)
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.HEIGHT)

    def _capture_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def start(self):
        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()
