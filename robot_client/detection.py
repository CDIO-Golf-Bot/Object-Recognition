# detection.py
#
# Run YOLO on each frame, detect balls/robot/obstacles,
# convert to world coordinates, and pass along to navigation.

import random
import cv2
import numpy as np
from ultralytics import YOLO
from queue import Empty

from robot_client import config, navigation, robot_comm
from robot_client.navigation import planner
from robot_client import config as client_config

# === Global State ===
yolo_model = YOLO("weights_v4.pt")  # Adjust path if needed
ball_positions_cm = []
obstacles = set()
class_colors = {}


def process_frames(frame_queue, output_queue, stop_event):
    global ball_positions_cm, obstacles

    while not stop_event.is_set():
        # Grab either (frame, timestamp) or raw frame
        try:
            item = frame_queue.get(timeout=0.02)
        except Empty:
            continue

        if isinstance(item, tuple) and len(item) == 2:
            frame, capture_ts = item
        else:
            frame = item
            capture_ts = time.time()

        original = frame.copy()

        # — ArUco detection & update shared state —
        aruco_corners, aruco_ids = navigation.detect_aruco(original)
        if aruco_ids is not None:
            for idx, marker_id in enumerate(aruco_ids.flatten()):
                if marker_id != config.ARUCO_MARKER_ID:
                    continue
                pts = aruco_corners[idx][0]
                cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())
                real = navigation.pixel_to_cm(int(round(cx)), int(round(cy)))
                if real:
                    x_cm, y_cm = real
                    heading_deg = navigation.compute_aruco_heading(pts)

                    # Update planner and heading
                    planner.robot_position_cm = (x_cm, y_cm)
                    client_config.ROBOT_HEADING = float(heading_deg)
                    robot_comm.send_pose(x_cm, y_cm, heading_deg)
                break

        # — YOLO inference —
        results = yolo_model(original, verbose=False)
        detections = results[0].boxes.data.tolist()

        ball_positions_cm.clear()
        new_obstacles = set()

        for x1, y1, x2, y2, conf, cls_id in detections:
            lbl = results[0].names[int(cls_id)].lower()
            cx_pix, cy_pix = int((x1 + x2) / 2), int((y1 + y2) / 2)
            is_robot = 'robot' in lbl

            if not is_robot:
                color = class_colors.setdefault(lbl, (
                    random.randint(0,255),
                    random.randint(0,255),
                    random.randint(0,255)
                ))
                cv2.rectangle(original, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(original, lbl, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            real = navigation.pixel_to_cm(cx_pix, cy_pix)
            if real:
                gx, gy = navigation.cm_to_grid_coords(real[0], real[1])

                if 'ball' in lbl and not (
                    config.IGNORED_AREA['x_min'] <= real[0] <= config.IGNORED_AREA['x_max'] and
                    config.IGNORED_AREA['y_min'] <= real[1] <= config.IGNORED_AREA['y_max']
                ):
                    ball_positions_cm.append((real[0], real[1], lbl, cx_pix, cy_pix))
                elif is_robot and planner.robot_position_cm is None:
                    planner.robot_position_cm = real
                else:
                    new_obstacles.add((gx, gy))

        obstacles |= navigation.get_expanded_obstacles(new_obstacles)

        # — Draw grid and route —
        frame_grid  = navigation.draw_metric_grid(original)
        frame_route = navigation.draw_full_route(frame_grid, ball_positions_cm)

        # Push (processed_frame, original_timestamp) for display
        try:
            output_queue.put((frame_route, capture_ts), timeout=0.02)
        except Exception:
            pass

    print("🖥️ process_frames exiting")
