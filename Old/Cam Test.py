import cv2
import numpy as np

# Ask the user for the camera index
cam_index = int(input("Enter camera index (default is 1): ") or 1)
cap = cv2.VideoCapture(cam_index)

if not cap.isOpened():
    print(f"Error: Could not open camera {cam_index}")
    exit()

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

# Capture the first frame to select points
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
        print("Selection canceled.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

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

# Process the video feed
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Apply perspective warp
    warped = cv2.warpPerspective(frame, H, (width, height))

    # Show both original and transformed frames
    cv2.imshow("Original Video", frame)
    cv2.imshow("Top-Down View", warped)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
