import cv2
import numpy as np

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

# Open the webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Wait for the first frame to select points
ret, frame = cap.read()
if not ret:
    print("Error: Could not capture video.")
    cap.release()
    exit()

cv2.imshow("Select 4 Points", frame)
cv2.setMouseCallback("Select 4 Points", get_points)

# Wait until 4 points are selected
while not capturing:
    cv2.imshow("Select 4 Points", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()  # Close the selection window

# Convert points to a NumPy array
src_points = np.array(points, dtype=np.float32)

# Define the four corresponding points in the transformed (top-down) view
width, height = 400, 600  # Adjust output size
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
    def __init__(self, 
                 weights_path="C:/Users/balde/Desktop/yolo/yolov4-tiny.weights", 
                 cfg_path="C:/Users/balde/Desktop/yolo/yolov4-tiny.cfg", 
                 classes_path="C:/Users/balde/Desktop/yolo/coco.names"):
        print("Loading Object Detection")
        print("Running OpenCV DNN with YOLOv4-Tiny")
        self.nmsThreshold = 0.5  # Non-Maximum Suppression Threshold
        self.confThreshold = 0.4  # Confidence Threshold
        self.image_size = 416  # YOLOv4-Tiny works best with 416x416

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
                self.classes = [class_name.strip() for class_name in file_object.readlines()]
            print(f"Loaded {len(self.classes)} class names.")
        except Exception as e:
            print(f"Error loading class names: {e}")

    def detect(self, frame):
        if self.model is None:  
            print("Error: Model is not loaded. Cannot perform detection.")
            return [], [], []

        # Perform detection
        detections = self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)

        if len(detections) == 3:  # Ensure we have the correct structure
            classes, confidences, boxes = detections
        else:
            print("Detection returned unexpected structure. Skipping frame.")
            return [], [], []

        return list(classes), list(confidences), list(boxes)  # Convert to lists

# ✅ Move initialization OUTSIDE the class
object_detector = ObjectDetection()

# Process the video feed
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply perspective warp
    warped = cv2.warpPerspective(frame, H, (width, height))

    # Detect objects in the transformed frame
    classes, confidences, boxes = object_detector.detect(warped)

    # Draw bounding boxes for all detected objects
    for class_id, confidence, box in zip(classes, confidences, boxes):
        color = object_detector.colors[class_id]
        label = object_detector.classes[class_id]  # Get object name
        cv2.rectangle(warped, box, color.tolist(), 2)
        cv2.putText(warped, f"{label}: {confidence:.2f}", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)

    # Show original and transformed frames
    cv2.imshow("Original Video", frame)
    cv2.imshow("Top-Down Object Detection", warped)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
