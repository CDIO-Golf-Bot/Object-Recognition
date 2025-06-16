# detection.py
#
# Run YOLO on each frame, detect balls/robot/obstacles,
# convert to world coordinates, and pass along to navigation.
# Run YOLO on each frame, detect balls/robot/obstacles,
# convert to world coordinates, and pass along to navigation.

import random
import cv2
import numpy as np
from ultralytics import YOLO
from queue import Empty

from robot_client import config
from robot_client import navigation
from robot_client import robot_comm
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
        try:
            frame = frame_queue.get(timeout=0.02)
        except Empty:
            continue

        original = frame.copy()

        # — ArUco detection & update shared state —
        # Directly use the exact pixel center of the marker
        aruco_corners, aruco_ids = navigation.detect_aruco(original)
        if aruco_ids is not None:
            for idx, marker_id in enumerate(aruco_ids.flatten()):
                if marker_id != config.ARUCO_MARKER_ID:
                    continue
                # Sub-pixel refined corners returned by detect_aruco
                pts = aruco_corners[idx][0]
                cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())

                # Map pixel center directly to world cm coords
                real = navigation.pixel_to_cm(int(round(cx)), int(round(cy)))
                if real:
                    x_cm, y_cm = real
                    heading_deg = navigation.compute_aruco_heading(pts)

                    # 1️⃣ Offset start to 'front' of tag if configured
                    #offset = getattr(config, 'START_OFFSET_CM', 0.0)
                    #if offset != 0.0:
                    #    theta = np.radians(heading_deg)
                    #    dx = offset * np.cos(theta)
                    #    dy = -offset * np.sin(theta)
                    #    x_cm += dx
                    #    y_cm += dy

                    # 2️⃣ Write into planner state so compute_best_route sees it:
                    planner.robot_position_cm = (x_cm, y_cm)

                    # 3️⃣ Keep your config heading in sync:
                    client_config.ROBOT_HEADING = float(heading_deg)

                    # 4️⃣ Send to robot server if desired:
                    robot_comm.send_pose(x_cm, y_cm, heading_deg)
                break

        # — YOLO inference —
        results    = yolo_model(original, verbose=False)
        detections = results[0].boxes.data.tolist()

        ball_positions_cm.clear()
        new_obstacles = set()

        for x1, y1, x2, y2, conf, cls_id in detections:
            lbl = results[0].names[int(cls_id)].lower()
            cx_pix, cy_pix = int((x1 + x2) / 2), int((y1 + y2) / 2)

            is_robot = "robot" in lbl
            if not is_robot:
                # draw other detections
                color = class_colors.setdefault(lbl, (
                    random.randint(0,255),
                    random.randint(0,255),
                    random.randint(0,255)
                ))
                cv2.rectangle(original, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(original, lbl, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # map center pixel to real‑world
            real = navigation.pixel_to_cm(cx_pix, cy_pix)
            if real:
                gx, gy = navigation.cm_to_grid_coords(real[0], real[1])

            if "ball" in lbl and real:
                if not (
                    config.IGNORED_AREA['x_min'] <= real[0] <= config.IGNORED_AREA['x_max'] and
                    config.IGNORED_AREA['y_min'] <= real[1] <= config.IGNORED_AREA['y_max']
                ):
                    ball_positions_cm.append((real[0], real[1], lbl, cx_pix, cy_pix))

            elif is_robot and navigation.robot_position_cm is None:
                navigation.robot_position_cm = real

            elif real:
                new_obstacles.add((gx, gy))

        obstacles |= navigation.get_expanded_obstacles(new_obstacles)

        # — Red‑cross obstacle detection —
        # (Unchanged)

        # — Draw grid and route —
        frame_grid  = navigation.draw_metric_grid(original)
        frame_route = navigation.draw_full_route(frame_grid, ball_positions_cm)

        try:
            output_queue.put(frame_route, timeout=0.02)
        except:
            pass

    print("🖥️ process_frames exiting")
