# navigation_helpers.py
# Add these functions into robot_client/navigation/__init__.py or a new module under navigation.

import cv2
import numpy as np
from . import calibration as cal


def detect_aruco(frame):
    """
    Detect ArUco markers with sub-pixel corner refinement.
    Returns (corners, ids) arrays as output by cv2.aruco.
    """
    # Undistort if calibration available
    if hasattr(cal, 'K') and cal.K is not None and hasattr(cal, 'distCoeffs') and cal.distCoeffs is not None:
        frame = cv2.undistort(frame, cal.K, cal.distCoeffs)
    # Prepare dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    params     = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(aruco_dict, params)

    # Convert to grayscale for detection and refinement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(frame)

    # Refine corners if found
    if ids is not None:
        for i, pts in enumerate(corners):
            raw = pts[0].astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            refined = cv2.cornerSubPix(gray, raw, (5,5), (-1,-1), criteria)
            corners[i][0] = refined
        # Draw for visualization
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    return corners, ids


def compute_aruco_heading(pts):
    """
    Compute heading in degrees from top-left to top-right corners.
    pts should be a 4x2 array of corner positions.
    """
    dx = pts[1,0] - pts[0,0]
    dy = pts[1,1] - pts[0,1]
    angle_rad = np.arctan2(-dy, dx)
    return (np.degrees(angle_rad) + 360) % 360
