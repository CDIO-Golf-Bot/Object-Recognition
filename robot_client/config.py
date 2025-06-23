# config.py
#
# All the adjustable settings: camera, grid sizes, thresholds,
# network addresses, drawing colors, and other constants.


import numpy as np
import random

# === Roboflow / Model Config ===
ROBOFLOW_API_KEY = "7kMjalIwU9TqGmKM0g4i"
WORKSPACE_NAME   = "pingpong-fafrv"
PROJECT_NAME     = "newpingpongdetector"
VERSION          = 1

# === Camera Settings ===
CAMERA_INDEX   = 1
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
FRAMES_PER_SEC = 30
CAMERA_BRIGHTNESS = 70     # 70% of default
BUFFER_SIZE    = 1
SKIP_FRAMES = 1

# === Real-World Grid Dimensions ===
REAL_WIDTH_CM   = 167
REAL_HEIGHT_CM  = 122
GRID_SPACING_CM = 2
CAMERA_HEIGHT = 180             # cm from ground aprox
ARUCO_MARKER_HEIGHT = 15        # cm from ground aprox

PLANE_SCALE = CAMERA_HEIGHT / (CAMERA_HEIGHT - ARUCO_MARKER_HEIGHT)

ARUCO_MARKER_ID = 100
ARUCO_REFERENCE_POINT  = "center"
START_OFFSET_CM = 0.0

# border‐buffer around the entire field (in cm → grid cells)
BORDER_BUFFER_CM    = 13
BORDER_BUFFER_CELLS = int(np.ceil(BORDER_BUFFER_CM / GRID_SPACING_CM))

# path
ARRIVAL_THRESHOLD_CM = 7.0   # cm distance to goal to consider arrived
MAX_SEGMENT_TIME = 10.0          # Max seconds to wait for a segment to complete (skip if too long)


START_POINT_CM = (20, 20)
GOAL_A_CM      = (REAL_WIDTH_CM - 18, (REAL_HEIGHT_CM // 2))
GOAL_B_CM      = None

GOAL_RANGE = {
    'A': [GOAL_A_CM],
    'B': GOAL_B_CM
}

# === Obstacle Expansion ===
OBSTACLE_BUFFER_CM = 10
BUFFER_CELLS = int(np.ceil(OBSTACLE_BUFFER_CM / GRID_SPACING_CM))

# === Detection Thresholds ===
MAX_BALLS_TO_COLLECT = 3
CONFIDENCE_THRESHOLD = 0.50
OVERLAP_THRESHOLD    = 0.05
MIN_RED_AREA_PX      = 500
MAX_RED_AREA_CM2     = 400

IGNORED_AREA = {
    'x_min': 0, 'x_max': 0,
    'y_min': 0, 'y_max': 0
}
homography_matrix     = None
inv_homography_matrix = None

MIN_RED_AREA_PX      = 500
MAX_RED_AREA_CM2     = 400

# === Robot Config ===
ROBOT_IP      = "10.137.48.57"
ROBOT_PORT    = 12345
ROBOT_HEADING = 0.0

# === Drawing Options ===
OBSTACLE_DRAW_RADIUS_PX = 6
GRID_LINE_COLOR         = (100, 100, 100)
PATH_COLOR              = (0, 255, 255)
TEXT_COLOR              = (0, 255, 255)

# === Misc ===
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
