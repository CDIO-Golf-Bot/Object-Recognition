"""
calibration.py

Handle all camera ↔ world calibration tasks:
  • Gather four user-clicked corner points to compute homography.
  • Detect ArUco marker #100 to estimate robot position & heading in cm.
  • Build and cache a semi-transparent metric grid overlay.
"""


import cv2
import numpy as np
import config

# === Calibration State ===
homography_matrix = None
inv_homography_matrix = None
calibration_points = []
grid_overlay = None


def get_aruco_robot_position_and_heading(frame):
    """
    Detect the ArUco marker #100, draw it on the frame,
    and return (x_cm, y_cm, heading_deg) in real-world coords, or None.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id != 100:
                continue
            pts = corners[idx][0]  # 4x2 array
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            angle_rad = np.arctan2(-dy, dx)
            heading_deg = (np.degrees(angle_rad) + 360) % 360
            cx, cy = pts[:,0].mean(), pts[:,1].mean()
            if inv_homography_matrix is not None:
                pt = np.array([[[cx, cy]]], dtype="float32")
                real_pt = cv2.perspectiveTransform(pt, inv_homography_matrix)[0][0]
                return real_pt[0], real_pt[1], heading_deg
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

    # create a blank cm‐resolution canvas
    canvas = np.zeros((config.REAL_HEIGHT_CM+1, config.REAL_WIDTH_CM+1, 3), dtype=np.uint8)
    for x in range(0, config.REAL_WIDTH_CM+1, config.GRID_SPACING_CM):
        cv2.line(canvas, (x, 0), (x, config.REAL_HEIGHT_CM), config.GRID_LINE_COLOR, 1)
    for y in range(0, config.REAL_HEIGHT_CM+1, config.GRID_SPACING_CM):
        cv2.line(canvas, (0, y), (config.REAL_WIDTH_CM, y), config.GRID_LINE_COLOR, 1)

    w_px, h_px = config.FRAME_WIDTH, config.FRAME_HEIGHT
    grid_overlay = cv2.warpPerspective(canvas, homography_matrix, (w_px, h_px), flags=cv2.INTER_LINEAR)
    print("🗺️ Cached grid overlay.")
