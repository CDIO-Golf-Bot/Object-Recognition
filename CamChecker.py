import cv2

for i in range(5):  # Test camera indices 0 to 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()

# Ask user for the camera index
cam_index = int(input("Enter camera index (default is 1 for USB cam): ") or 1)

cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print(f"Error: Could not open camera {cam_index}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow("USB Camera Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()