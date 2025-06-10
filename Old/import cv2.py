import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load YOLOv8 model
model.export(format="onnx")  # Export to ONNX format
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

# Load class labels (e.g., "person", "ball", etc.)
with open("coco.names", "r") as f:  
    classes = [line.strip() for line in f.readlines()]

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Ensure quadrilateral shape
            src_points = np.array([point[0] for point in approx], dtype=np.float32)

            # Sort points correctly: top-left, top-right, bottom-left, bottom-right
            sum_pts = src_points.sum(axis=1)
            diff_pts = np.diff(src_points, axis=1)

            top_left = src_points[np.argmin(sum_pts)]
            bottom_right = src_points[np.argmax(sum_pts)]
            top_right = src_points[np.argmin(diff_pts)]
            bottom_left = src_points[np.argmax(diff_pts)]

            src_points = np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)

            # Define output size
            width, height = 400, 600
            dst_points = np.array([
                [0, 0],
                [width, 0],
                [0, height],
                [width, height]
            ], dtype=np.float32)

            # Compute perspective transform
            H = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(frame, H, (width, height))

            # Object Detection (YOLO)
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward(layer_names)

            for output in detections:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:  # Confidence threshold
                        box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        label = f"{classes[class_id]}: {confidence:.2f}"
                        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show both original and top-down view
            cv2.imshow("Original Video", frame)
            cv2.imshow("Top-Down View", warped)
            break  # Process only the first detected field

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
