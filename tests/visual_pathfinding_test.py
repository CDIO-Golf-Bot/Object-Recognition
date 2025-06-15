import cv2
import numpy as np
import sys
import os

from robot_client import config
from robot_client.navigation import calibration, grid_utils as gu, planner

# === Settings ===
SIM_WIDTH_PX = 900
SIM_HEIGHT_PX = 600

# === Global State ===
ball_positions = []
def mouse_callback(event, x, y, flags, param):
    cm = gu.pixel_to_cm(x, y)
    if cm is None:
        return
    x_cm, y_cm = cm

    if event == cv2.EVENT_LBUTTONDOWN:
        if planner.robot_position_cm is None:
            planner.robot_position_cm = (x_cm, y_cm)
            print(f"🤖 Robot set at ({x_cm:.1f}, {y_cm:.1f})")
        else:
            ball_positions.append((x_cm, y_cm))
            print(f"⚽ Ball added at ({x_cm:.1f}, {y_cm:.1f})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if ball_positions:
            dist = lambda b: (b[0] - x_cm)**2 + (b[1] - y_cm)**2
            closest = min(ball_positions, key=dist)
            ball_positions.remove(closest)
            print(f"❌ Ball removed at ({closest[0]:.1f}, {closest[1]:.1f})")
        elif planner.robot_position_cm is not None:
            print("❌ Robot removed")
            planner.robot_position_cm = None

def render_frame():
    # Draw background
    frame = np.zeros((SIM_HEIGHT_PX, SIM_WIDTH_PX, 3), dtype=np.uint8)
    frame = gu.draw_metric_grid(frame)

    # Use existing planner's drawing
    formatted = [(x, y, "test_ball", int(x), int(y)) for (x, y) in ball_positions]
    frame = planner.draw_full_route(frame, formatted)

    # Draw annotated balls
    for x_cm, y_cm in ball_positions:
        pt = np.array([[[x_cm, y_cm]]], dtype="float32")
        px, py = cv2.perspectiveTransform(pt, calibration.homography_matrix)[0][0]
        cv2.circle(frame, (int(px), int(py)), 6, (0, 255, 0), -1)
        label = f"({x_cm:.1f},{y_cm:.1f})"
        cv2.putText(frame, label, (int(px) + 6, int(py) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw annotated robot (if present)
    if planner.robot_position_cm:
        x_cm, y_cm = planner.robot_position_cm
        pt = np.array([[[x_cm, y_cm]]], dtype="float32")
        px, py = cv2.perspectiveTransform(pt, calibration.homography_matrix)[0][0]
        cv2.circle(frame, (int(px), int(py)), 6, (0, 0, 255), -1)
        label = f"ROBOT ({x_cm:.1f},{y_cm:.1f})"
        cv2.putText(frame, label, (int(px) + 6, int(py) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return frame

def test_visual_pathfinding():
    config.FRAME_WIDTH = SIM_WIDTH_PX
    config.FRAME_HEIGHT = SIM_HEIGHT_PX

    # Setup homography
    src_pts = np.array([
        [0, 0],
        [config.REAL_WIDTH_CM, 0],
        [config.REAL_WIDTH_CM, config.REAL_HEIGHT_CM],
        [0, config.REAL_HEIGHT_CM]
    ], dtype="float32")

    dst_pts = np.array([
        [50, 50],
        [SIM_WIDTH_PX - 50, 50],
        [SIM_WIDTH_PX - 50, SIM_HEIGHT_PX - 50],
        [50, SIM_HEIGHT_PX - 50]
    ], dtype="float32")

    calibration.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    calibration.inv_homography_matrix = np.linalg.inv(calibration.homography_matrix)
    calibration.create_and_cache_grid_overlay()

    # Setup window
    cv2.namedWindow("Pathfinding Test")
    cv2.setMouseCallback("Pathfinding Test", mouse_callback)
    print("🖱️ Left-click: place robot, then balls. Right-click: remove ball or robot.")

    while True:
        frame = render_frame()
        cv2.imshow("Pathfinding Test", frame)
        key = cv2.waitKey(20)
        if key == ord('q'):
            print("👋 Exiting.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_visual_pathfinding()
