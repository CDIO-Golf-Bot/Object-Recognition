import cv2
import numpy as np
import threading

# Global variables
points = []
capturing = False

# Mouse callback function to collect 4 points
def get_points(event, x, y, flags, param):
    global points, capturing
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        points.append((x, y))
        print(f"Point {len(points)}: {x}, {y}")
        if len(points) == 4:
            capturing = True  # Start processing frames

# Multi-threaded Video Capture Class
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# Start Video Stream
video_stream = VideoStream()

# Wait for the first frame
while video_stream.read() is None:
    pass

frame = video_stream.read()
cv2.imshow("Select 4 Points", frame)
cv2.setMouseCallback("Select 4 Points", get_points)

while not capturing:
    cv2.imshow("Select 4 Points", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video_stream.stop()
        exit()

cv2.destroyAllWindows()

# Convert points to NumPy array
src_points = np.array(points, dtype=np.float32)

# Define the four corresponding points in the transformed (top-down) view
width, height = 400, 600

dst_points = np.array([
    [0, 0],
    [width, 0],
    [0, height],
    [width, height]
], dtype=np.float32)

# Compute the perspective transformation matrix
H = cv2.getPerspectiveTransform(src_points, dst_points)

# Object Detection Class
class ObjectDetection:
    def __init__(self, weights_path="C:/Users/balde/Desktop/yolo/yolov4.weights", 
                 cfg_path="C:/Users/balde/Desktop/yolo/yolov4.cfg", 
                 classes_path="C:/Users/balde/Desktop/yolo/coco.names"):
        print("Loading Object Detection")
        self.nmsThreshold = 0.5
        self.confThreshold = 0.6
        self.image_size = 608

        try:
            net = cv2.dnn.readNet(weights_path, cfg_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names(classes_path)
        self.colors = np.random.uniform(0, 255, size=(80, 3))
        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path):
        try:
            with open(classes_path, "r") as file_object:
                self.classes = [line.strip() for line in file_object.readlines()]
            print(f"Loaded {len(self.classes)} class names.")
        except Exception as e:
            print(f"Error loading class names: {e}")

    def detect(self, frame):
        if self.model is None:
            return [], [], []

        classes, confidences, boxes = self.model.detect(frame, 
                                                         nmsThreshold=self.nmsThreshold, 
                                                         confThreshold=self.confThreshold)
        
        filtered_classes = []
        filtered_confidences = []
        filtered_boxes = []
        
        for class_id, confidence, box in zip(classes, confidences, boxes):
            label = self.classes[class_id]
            if label in ["sports ball", "car"]:
                filtered_classes.append(class_id)
                filtered_confidences.append(confidence)
                filtered_boxes.append(box)

        return filtered_classes, filtered_confidences, filtered_boxes

# Initialize Object Detection
object_detector = ObjectDetection()

# Process the video feed
while True:
    frame = video_stream.read()
    if frame is None:
        break

    # Apply perspective warp
    warped = cv2.warpPerspective(frame, H, (width, height))

    # Detect objects in the transformed frame
    classes, confidences, boxes = object_detector.detect(warped)

    # Draw bounding boxes
    for class_id, confidence, box in zip(classes, confidences, boxes):
        color = object_detector.colors[class_id]
        label = object_detector.classes[class_id]
        cv2.rectangle(warped, box, color.tolist(), 2)
        cv2.putText(warped, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)

    # Show original and transformed frames
    cv2.imshow("Original Video", frame)
    cv2.imshow("Top-Down Object Detection", warped)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.stop()
cv2.destroyAllWindows()
