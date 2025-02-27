import cv2
import threading

# Global variable to store available cameras
available_cams = []

# Function to check if a camera is available
def check_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"Camera {index} is available")
        available_cams.append(index)
    cap.release()

# Function to check cameras in parallel using threading
def check_all_cameras():
    threads = []
    for i in range(3):  # Check for the first 2 camera indices (adjust as needed)
        thread = threading.Thread(target=check_camera, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("Available cameras:", available_cams)

# Run the camera check in a separate thread
check_all_cameras()

# Ask the user for the camera index
cam_index = int(input("Enter camera index (default is 1 for USB cam): ") or 1)

# Open the selected camera
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: Could not open camera {cam_index}")
    exit()

print("Checking camera properties...")
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Camera {cam_index} properties:")
print(f"Frame Width: {frame_width}")
print(f"Frame Height: {frame_height}")
print(f"FPS: {fps}")

# Set resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Start capturing frames and display the feed
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
