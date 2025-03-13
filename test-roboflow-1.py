from roboflow import Roboflow
import cv2

# Initialize Roboflow with your API key
rf = Roboflow(api_key="7kMjalIwU9TqGmKM0g4i")
project = rf.workspace().project("CDIO")
model = project.version("YOUR_MODEL_VERSION").model

# Start video capture
cap = cv2.VideoCapture(0)

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
        cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}", (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Live Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
