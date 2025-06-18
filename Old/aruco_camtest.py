import cv2
import numpy as np

# Load ArUco dictionary and detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit()

print("📷 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect markers
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            if marker_id != 100:
                continue  # Skip other markers

            # Extract corner points
            pts = corners[i][0]  # 4x2 array
            pt1 = pts[0]  # Top-left
            pt2 = pts[1]  # Top-right

            # Compute heading angle
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            angle_rad = np.arctan2(-dy, dx)  # Negate dy to match screen coordinates
            angle_deg = (np.degrees(angle_rad) + 360) % 360  # Convert to 0–360°

            # Display heading
            cv2.putText(frame, f"Heading: {angle_deg:.1f}°",
                        (int(pt1[0]), int(pt1[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Heading Only", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
