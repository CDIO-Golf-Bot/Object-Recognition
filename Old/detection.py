from roboflow import Roboflow
import cv2
from queue import Queue

class RoboFlowDetector:
    """
    Handles camera capture and object detection using a Roboflow model.
    """
    def __init__(self, cam_index=0):
        # Initialize Roboflow
        self.rf = Roboflow(api_key="7kMjalIwU9TqGmKM0g4i")
        self.model = (
            self.rf
                .workspace("pingpong-fafrv")
                .project("pingpongdetector-rqboj")
                .version(2)
                .model
        )

        # Camera setup
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Threading queue
        self.frame_queue = Queue(maxsize=1)

    def capture_frames(self):
        """Continuously read frames from camera and enqueue."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def process_frames(self, callback):
        """
        Process frames: run detection and call callback(frame, detections).
        Detections: list of dicts with x, y, width, height, class.
        """
        while True:
            frame = self.frame_queue.get()
            resized = cv2.resize(frame, (416, 416))
            preds = self.model.predict(resized, confidence=30, overlap=20).json()
            detections = preds.get("predictions", [])

            h, w = frame.shape[:2]
            sx, sy = w / 416.0, h / 416.0
            for det in detections:
                det["x"] *= sx
                det["y"] *= sy
                det["width"] *= sx
                det["height"] *= sy

            callback(frame, detections)

    def release(self):
        """Release camera resources."""
        self.cap.release()