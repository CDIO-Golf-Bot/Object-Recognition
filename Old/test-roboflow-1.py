from roboflow import Roboflow
import cv2
import random

# Initialize Roboflow
rf = Roboflow(api_key="7kMjalIwU9TqGmKM0g4i")
project = rf.workspace("pingpong-fafrv").project("pingpongdetector-rqboj")
model = project.version(1).model  # Ensure this version exists

# Define colors for different classes
class_colors = {}

# Start video capture
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    predictions = model.predict(frame, confidence=40, overlap=30).json()

    # Process predictions (e.g., draw bounding boxes)
    for pred in predictions['predictions']:
        x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
        label = pred['class']
        confidence = pred['confidence']

        # Assign a unique color to each class
        if label not in class_colors:
            class_colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        color = class_colors[label]

        # Draw bounding box with class-specific color
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x - w // 2, y - h // 2 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow("Live Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
