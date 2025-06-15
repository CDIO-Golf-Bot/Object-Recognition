import cv2
import numpy as np
import sys
import os

# Add project root to sys.path so imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "package"))
sys.path.append(project_root)

import config
import navigation.calibration as calibration
import navigation.grid_utils as gu
import navigation.planner as planner
from navigation.grid_utils import pixel_to_cm

SIM_WIDTH_PX = 900
SIM_HEIGHT_PX = 600

# State
robot_set = False
ball_positions = []
def mouse_callback(event, x, y, flags, param):
    global robot_set, ball_positions

    cm = pixel_to_cm(x, y)
    if cm is None:
        return
    x_cm, y_cm = cm

    if event == cv2.EVENT_LBUTTONDOWN:
        if not robot_set:
            planner.robot_position_cm = (x_cm, y_cm)
            robot_set = True
            print(f"🤖 Robot set at ({x_cm:.1f}, {y_cm:.1f})")
        else:
            ball_positions.append((x_cm, y_cm))
            print(f"⚽ Ball added at ({x_cm:.1f}, {y_cm:.1f})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if ball_positions:
            # Remove closest ball
            dist = lambda b: (b[0] - x_cm)**2 + (b[1] - y_cm)**2
            closest = min(ball_positions, key=dist)
            ball_positions.remove(closest)
            print(f"❌ Ball removed at ({closest[0]:.1f}, {closest[1]:.1f})")
        elif robot_set:
            planner.robot_position_cm = None
            robot_set = False
            print("❌ Robot removed")

def render_frame():
    frame = np.zeros((SIM_HEIGHT_PX, SIM_WIDTH_PX, 3), dtype=np.uint8)
    frame = gu.draw_metric_grid(frame)

    formatted_balls = [(x, y, "test_ball", int(x), int(y)) for (x, y) in ball_positions]
    return planner.draw_full_route(frame, formatted_balls)

def test_visual_pathfinding():
    # Step 1: Patch config with simulation canvas size
    config.FRAME_WIDTH = SIM_WIDTH_PX
    config.FRAME_HEIGHT = SIM_HEIGHT_PX

    # Step 2: Set up hardcoded homography
    src_pts = np.array([
        [0, 0],
        [config.REAL_WIDTH_CM, 0],
        [config.REAL_WIDTH_CM, config.REAL_HEIGHT_CM],
        [0, config.REAL_HEIGHT_CM]
    ], dtype="float32")

    dst_pts = np.array([
        [0, 0],
        [SIM_WIDTH_PX, 0],
        [SIM_WIDTH_PX, SIM_HEIGHT_PX],
        [0, SIM_HEIGHT_PX]
    ], dtype="float32")

    calibration.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    calibration.inv_homography_matrix = np.linalg.inv(calibration.homography_matrix)

    # Step 3: Build the grid overlay at the correct resolution
    calibration.create_and_cache_grid_overlay()

    # Step 4: GUI
    cv2.namedWindow("Pathfinding Test")
    cv2.setMouseCallback("Pathfinding Test", mouse_callback)

    print("🖱️ Left-click: place robot (first), then balls. Right-click: remove closest ball.")

    while True:
        frame = render_frame()
        cv2.imshow("Pathfinding Test", frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            print("👋 Exiting test.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_visual_pathfinding()
