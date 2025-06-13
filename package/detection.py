import random
import cv2
import numpy as np
from ultralytics import YOLO
from queue import Empty
import navigation

from config import (
    MAX_BALLS_TO_COLLECT, IGNORED_AREA,
    REAL_WIDTH_CM, REAL_HEIGHT_CM,
    MIN_RED_AREA_PX, MAX_RED_AREA_CM2,
    GRID_SPACING_CM
)
from navigation import (
    pixel_to_cm, cm_to_grid_coords,
    get_expanded_obstacles,
    draw_metric_grid, draw_full_route,
    get_aruco_robot_position_and_heading
)

# === Global State ===
yolo_model = YOLO("weights_v3.pt")  # Adjust path if needed

ball_positions_cm = []
robot_position_cm = None
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

        aruco = get_aruco_robot_position_and_heading(original)
        if aruco:
            x_cm, y_cm, heading_deg = aruco
            # update local module state (if you need it here)
            ball_positions_cm  # no-op just to silence unused-warning
            # push into the shared navigation state:
            navigation.robot_position_cm = (x_cm, y_cm)
            navigation.ROBOT_HEADING    = heading_deg

        results = yolo_model(original, verbose=False)
        detections = results[0].boxes.data.tolist()

        ball_positions_cm.clear()
        new_obstacles = set()

        for x1, y1, x2, y2, conf, cls_id in detections:
            lbl = results[0].names[int(cls_id)].lower()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if "ball" in lbl:
                color = (200, 200, 255) if "white" in lbl else (0, 165, 255)
            else:
                color = class_colors.setdefault(lbl, (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                ))

            cv2.rectangle(original, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(original, lbl, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            real = pixel_to_cm(cx, cy)
            if real:
                gx, gy = cm_to_grid_coords(real[0], real[1])

            if "ball" in lbl:
                if real and not (
                    IGNORED_AREA['x_min'] <= real[0] <= IGNORED_AREA['x_max'] and
                    IGNORED_AREA['y_min'] <= real[1] <= IGNORED_AREA['y_max']
                ):
                    ball_positions_cm.append((real[0], real[1], lbl, cx, cy))

            elif "robot" in lbl:
                if real:
                    robot_position_cm = real

            else:
                if real:
                    new_obstacles.add((gx, gy))

        obstacles |= get_expanded_obstacles(new_obstacles)

        # RED CROSS DETECTION
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt, best_area = None, 0
        px_per_x = original.shape[1] / REAL_WIDTH_CM
        px_per_y = original.shape[0] / REAL_HEIGHT_CM

        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < MIN_RED_AREA_PX:
                continue
            x_r, y_r, w_r, h_r = cv2.boundingRect(cnt)
            area_cm = (w_r / px_per_x) * (h_r / px_per_y)
            if area_cm > MAX_RED_AREA_CM2:
                continue
            if area_px > best_area:
                best_cnt, best_area = cnt, area_px

        if best_cnt is not None:
            bx, by, bw, bh = cv2.boundingRect(best_cnt)
            for sx in range(bx, bx + bw, 10):
                for sy in range(by, by + bh, 10):
                    if cv2.pointPolygonTest(best_cnt, (sx, sy), False) >= 0:
                        real = pixel_to_cm(sx, sy)
                        if real:
                            gx, gy = cm_to_grid_coords(real[0], real[1])
                            if 0 <= gx <= REAL_WIDTH_CM // GRID_SPACING_CM and \
                               0 <= gy <= REAL_HEIGHT_CM // GRID_SPACING_CM:
                                obstacles.add((gx, gy))

        # DRAW GRID + ROUTE
        frame_grid = draw_metric_grid(original)
        frame_route = draw_full_route(frame_grid, ball_positions_cm)

        try:
            output_queue.put(frame_route, timeout=0.02)
        except:
            pass

    print("🖥️ process_frames exiting")
