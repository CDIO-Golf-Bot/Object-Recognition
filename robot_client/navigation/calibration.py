import cv2
import numpy as np
from robot_client import config

# === Calibration State ===
homography_matrix = None
inv_homography_matrix = None
calibration_points = []
grid_overlay = None


def get_aruco_robot_position_and_heading(frame):
    """
    Detect the ArUco marker (config.ARUCO_MARKER_ID), draw it on the frame,
    and return (x_cm, y_cm, heading_deg) in real-world coords, or None.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id != config.ARUCO_MARKER_ID:
                continue

            # Extract the four corner points of the detected marker
            pts = corners[idx][0]  # 4x2 array: [top-left, top-right, bottom-right, bottom-left]

            # Compute heading using the vector from top-left to top-right
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            angle_rad = np.arctan2(-dy, dx)
            heading_deg = (np.degrees(angle_rad) + 360) % 360

            # Choose reference point on the marker (center, bottom-right, or top-left)
            if config.ARUCO_REFERENCE_POINT == "center":
                cx = pts[:, 0].mean()
                cy = pts[:, 1].mean()
            elif config.ARUCO_REFERENCE_POINT == "bottom_right":
                cx, cy = pts[2]
            elif config.ARUCO_REFERENCE_POINT == "top_left":
                cx, cy = pts[0]
            else:
                # Fallback to center if config is invalid
                cx = pts[:, 0].mean()
                cy = pts[:, 1].mean()

            # Warp the chosen point into real-world (table-plane) coordinates
            if inv_homography_matrix is not None:
                pt = np.array([[[cx, cy]]], dtype="float32")
                real_pt = cv2.perspectiveTransform(pt, inv_homography_matrix)[0][0]

                # Correct for marker height above the table
                H = config.CAMERA_HEIGHT
                h = config.ARUCO_MARKER_HEIGHT
                s = (H - h) / H
                real_x = real_pt[0] * s
                real_y = real_pt[1] * s

                return real_x, real_y, heading_deg
    return None


def click_to_set_corners(event, x, y, flags, param):
    """
    Mouse callback: left-click to add a corner (up to 4);
    right-click to remove the most recent corner.
    After 4 points, compute homography and cache a grid overlay.
    """
    global calibration_points, homography_matrix, inv_homography_matrix, grid_overlay

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append([x, y])
            print(f"Corner {len(calibration_points)} set: ({x}, {y})")
        if len(calibration_points) == 4 and homography_matrix is None:
            dst = np.array([
                [0, 0],
                [config.REAL_WIDTH_CM, 0],
                [config.REAL_WIDTH_CM, config.REAL_HEIGHT_CM],
                [0, config.REAL_HEIGHT_CM]
            ], dtype="float32")
            src = np.array(calibration_points, dtype="float32")
            homography_matrix = cv2.getPerspectiveTransform(dst, src)
            inv_homography_matrix = np.linalg.inv(homography_matrix)
            print("✅ Homography calculated.")
            create_and_cache_grid_overlay()

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Always allow removing the last corner
        if calibration_points:
            removed = calibration_points.pop()
            print(f"❌ Removed corner: {removed}")
        # Reset homography if we have fewer than 4 points
        if len(calibration_points) < 4:
            homography_matrix = None
            inv_homography_matrix = None
            grid_overlay = None


def create_and_cache_grid_overlay():
    """
    Build a transparent metric grid overlay (cm → pixel) and cache it.
    If no homography has been computed yet, do nothing.
    """
    global grid_overlay
    if homography_matrix is None:
        return

    # create a blank cm-resolution canvas
    canvas = np.zeros((config.REAL_HEIGHT_CM+1, config.REAL_WIDTH_CM+1, 3), dtype=np.uint8)
    for x in range(0, config.REAL_WIDTH_CM+1, config.GRID_SPACING_CM):
        cv2.line(canvas, (x, 0), (x, config.REAL_HEIGHT_CM), config.GRID_LINE_COLOR, 1)
    for y in range(0, config.REAL_HEIGHT_CM+1, config.GRID_SPACING_CM):
        cv2.line(canvas, (0, y), (config.REAL_WIDTH_CM, y), config.GRID_LINE_COLOR, 1)

    w_px, h_px = config.FRAME_WIDTH, config.FRAME_HEIGHT
    grid_overlay = cv2.warpPerspective(canvas, homography_matrix, (w_px, h_px), flags=cv2.INTER_LINEAR)
    print("🗺️ Cached grid overlay.")
