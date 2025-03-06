import cv2
import numpy as np

class ObjectDetection:
    def __init__(self, weights_path="C:/Users/balde/Desktop/Code/yolo/yolov4-tiny.weights", 
                 cfg_path="C:/Users/balde/Desktop/Code/yolo/yolov4-tiny.cfg"):
        print("Loading Object Detection")
        print("Running OpenCV DNN with YOLOv4-tiny")
        self.nmsThreshold = 0.5
        self.confThreshold = 0.6
        self.image_size = 416  # Lower resolution for faster processing

        # Load Network
        try:
            net = cv2.dnn.readNet(weights_path, cfg_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # Uncomment these lines if you have GPU support with CUDA:
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        # Use CPU if GPU not available:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="C:/Users/balde/Desktop/Code/yolo/coco.names"):
        try:
            with open(classes_path, "r") as file_object:
                for class_name in file_object.readlines():
                    class_name = class_name.strip()
                    self.classes.append(class_name)
            print(f"Loaded {len(self.classes)} class names.")
        except Exception as e:
            print(f"Error loading class names: {e}")
            return []
        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)

def run_object_detection():
    # Initialize the ObjectDetection class
    object_detector = ObjectDetection()

    # Open a video capture object (webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Detect objects in the current frame
        classes, confidences, boxes = object_detector.detect(frame)

        # Draw bounding boxes and class names on the detected objects
        for class_id, confidence, box in zip(classes, confidences, boxes):
            color = object_detector.colors[class_id]
            label = object_detector.classes[class_id]
            cv2.rectangle(frame, box, color.tolist(), 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the object detection
run_object_detection()
