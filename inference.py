from ultralytics import YOLO
import cv2

# Replace with the path to your downloaded weights
model = YOLO("weights.pt")  # e.g., "best.pt"

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Draw results
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
