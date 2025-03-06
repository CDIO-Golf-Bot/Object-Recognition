import cv2
import numpy as np

def main():
    # Start the camera feed
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Loop to continuously capture frames from the webcam
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Draw a red circle in the center of the frame (fixed position)
        cv2.circle(frame, (250, 250), 50, (0, 0, 255), -1)  # Red circle at (250, 250)

        # Display the frame with the circle
        cv2.imshow("Webcam Feed with Circle", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
