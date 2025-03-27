from roboflow import Roboflow
import cv2
import random
import threading
import time
import numpy as np
from queue import Queue
import heapq

# ✅ Initialize Roboflow
rf = Roboflow(api_key="7kMjalIwU9TqGmKM0g4i")
project = rf.workspace("pingpong-fafrv").project("pingpongdetector-rqboj")
model = project.version(1).model

class_colors = {}

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_queue = Queue(maxsize=1)
output_queue = Queue(maxsize=1)

skip_frames = 3
frame_count = 0

calibration_points = []
homography_matrix = None
real_width_cm = 180
real_height_cm = 120
grid_spacing_cm = 5

obstacles = set()
ball_positions_cm = []  # Each element is now (cx, cy, label)
start_point_cm = (20, 20)

goal_range = {
    'A': [(real_width_cm, y) for y in range(56, 65)],
    # 'B': [(real_width_cm - 5, y) for y in range(56, 65)]
}
selected_goal = 'A'
placing_goal = False

# Define an ignored area in cm (adjust these values as needed)
ignored_area = {
    'x_min': 50,
    'x_max': 100,
    'y_min': 50,
    'y_max': 100
}

def click_to_set_corners(event, x, y, flags, param):
    global calibration_points, homography_matrix, obstacles
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append([x, y])
            print(f"Corner {len(calibration_points)} set: ({x}, {y})")
        if len(calibration_points) == 4:
            dst_points = np.array([
                [0, 0],
                [real_width_cm, 0],
                [real_width_cm, real_height_cm],
                [0, real_height_cm]
            ], dtype="float32")
            src_points = np.array(calibration_points, dtype="float32")
            homography_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
            print("✅ Homography calculated.")
            mark_center_obstacle()
            ensure_outer_edges_walkable()
    elif event == cv2.EVENT_RBUTTONDOWN and homography_matrix is not None:
        pt_cm = pixel_to_cm(x, y)
        if pt_cm is not None:
            gx, gy = cm_to_grid_coords(pt_cm[0], pt_cm[1])
            if 0 <= gx <= real_width_cm // grid_spacing_cm and 0 <= gy <= real_height_cm // grid_spacing_cm:
                obstacles.symmetric_difference_update({(gx, gy)})
                print(f"🚧 Toggled obstacle at grid cell: ({gx}, {gy})")

def mark_center_obstacle():
    cx, cy = real_width_cm / 2, real_height_cm / 2
    half = 10
    for x_cm in range(int(cx - half), int(cx + half), grid_spacing_cm):
        for y_cm in range(int(cy - half), int(cy + half), grid_spacing_cm):
            gx, gy = cm_to_grid_coords(x_cm, y_cm)
            obstacles.add((gx, gy))
    print("🟥 Center obstacle (20x20 cm) added.")

def ensure_outer_edges_walkable():
    max_x = real_width_cm // grid_spacing_cm
    max_y = real_height_cm // grid_spacing_cm
    for gx in range(max_x + 1):
        obstacles.discard((gx, 0))
        obstacles.discard((gx, max_y))
    for gy in range(max_y + 1):
        obstacles.discard((0, gy))
        obstacles.discard((max_x, gy))
    print("✅ Outer edges cleared.")

def draw_metric_grid(frame):
    if homography_matrix is None:
        return frame
    overlay = frame.copy()
    for x_cm in range(0, real_width_cm + 1, grid_spacing_cm):
        pt1 = np.array([[[x_cm, 0]]], dtype="float32")
        pt2 = np.array([[[x_cm, real_height_cm]]], dtype="float32")
        pt1 = cv2.perspectiveTransform(pt1, homography_matrix)[0][0]
        pt2 = cv2.perspectiveTransform(pt2, homography_matrix)[0][0]
        cv2.line(overlay, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (100, 100, 100), 1)
    for y_cm in range(0, real_height_cm + 1, grid_spacing_cm):
        pt1 = np.array([[[0, y_cm]]], dtype="float32")
        pt2 = np.array([[[real_width_cm, y_cm]]], dtype="float32")
        pt1 = cv2.perspectiveTransform(pt1, homography_matrix)[0][0]
        pt2 = cv2.perspectiveTransform(pt2, homography_matrix)[0][0]
        cv2.line(overlay, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (100, 100, 100), 1)
    for (gx, gy) in obstacles:
        pt_cm = np.array([[[gx * grid_spacing_cm, gy * grid_spacing_cm]]], dtype='float32')
        pt_px = cv2.perspectiveTransform(pt_cm, homography_matrix)[0][0]
        cv2.circle(overlay, tuple(pt_px.astype(int)), 6, (0, 0, 255), -1)
    for label, pts in goal_range.items():
        for pt in pts:
            px = cv2.perspectiveTransform(np.array([[[pt[0], pt[1]]]], dtype="float32"), homography_matrix)[0][0]
            cv2.circle(overlay, tuple(px.astype(int)), 4, (0, 255, 0), -1)
        mid_y = (pts[0][1] + pts[-1][1]) // 2
        px = cv2.perspectiveTransform(np.array([[[pts[0][0], mid_y]]], dtype="float32"), homography_matrix)[0][0]
        cv2.putText(overlay, f"Goal {label}", tuple((px + 10).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    # 🔵 Start point marker
    pt_cm = np.array([[[start_point_cm[0], start_point_cm[1]]]], dtype="float32")
    pt_px = cv2.perspectiveTransform(pt_cm, homography_matrix)[0][0]
    cv2.circle(overlay, tuple(pt_px.astype(int)), 8, (255, 0, 0), -1)
    cv2.putText(overlay, "Start", tuple((pt_px + 10).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return overlay

def pixel_to_cm(px, py):
    if homography_matrix is None:
        return None
    pt = np.array([[[px, py]]], dtype="float32")
    inv_h = np.linalg.inv(homography_matrix)
    real_pt = cv2.perspectiveTransform(pt, inv_h)[0][0]
    return real_pt

def cm_to_grid_coords(x_cm, y_cm):
    return int(x_cm // grid_spacing_cm), int(y_cm // grid_spacing_cm)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        x, y = current
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (x + dx, y + dy)
            if (0 <= neighbor[0] <= real_width_cm // grid_spacing_cm and
                0 <= neighbor[1] <= real_height_cm // grid_spacing_cm and
                neighbor not in obstacles):
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
    return []

def draw_full_route(frame, ball_positions):
    if homography_matrix is None:
        return frame

    # Separate balls into non-orange and orange (using case-insensitive comparison)
    non_orange_balls = [ball for ball in ball_positions if ball[2].strip().lower() != "orange"]
    orange_balls = [ball for ball in ball_positions if ball[2].strip().lower() == "orange"]

    # Choose one orange ball if available (if multiple, adjust selection as needed)
    orange_ball = orange_balls[0] if orange_balls else None

    # Build the route starting at the start point
    route = [start_point_cm]
    current = start_point_cm
    non_orange_remaining = non_orange_balls.copy()
    while non_orange_remaining:
        next_ball = min(
            non_orange_remaining,
            key=lambda b: heuristic(cm_to_grid_coords(*current), cm_to_grid_coords(b[0], b[1]))
        )
        route.append((next_ball[0], next_ball[1]))
        non_orange_remaining.remove(next_ball)
        current = (next_ball[0], next_ball[1])

    # Append the orange ball last if it exists
    if orange_ball:
        route.append((orange_ball[0], orange_ball[1]))
        current = (orange_ball[0], orange_ball[1])

    # Now choose the goal candidate based on the last route point
    goal_candidates = goal_range.get(selected_goal)
    if not goal_candidates:
        return frame
    goal_cm = min(goal_candidates, key=lambda g: heuristic(cm_to_grid_coords(*current), cm_to_grid_coords(*g)))
    route.append(goal_cm)

    # Debug: print computed route (in centimeters)
    print("Computed route (in cm):", route)

    overlay = frame.copy()
    path_color = (0, 255, 255)
    total_cm = 0
    for i in range(len(route)-1):
        start_coords = cm_to_grid_coords(*route[i])
        end_coords = cm_to_grid_coords(*route[i+1])
        path = astar(start_coords, end_coords)
        prev_pt = None
        for gx, gy in path:
            pt_cm = np.array([[[gx * grid_spacing_cm, gy * grid_spacing_cm]]], dtype='float32')
            pt_px = cv2.perspectiveTransform(pt_cm, homography_matrix)[0][0]
            pt_px = tuple(pt_px.astype(int))
            if prev_pt is not None:
                cv2.line(overlay, prev_pt, pt_px, path_color, 3)
                total_cm += grid_spacing_cm
            prev_pt = pt_px
    cv2.putText(overlay, f"Total Path: {total_cm:.0f}cm to Goal {selected_goal}", 
                (10, overlay.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, path_color, 2)
    return overlay

def capture_frames():
    global frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % skip_frames == 0:
            if not frame_queue.full():
                frame_queue.put(frame)

def process_frames():
    global ball_positions_cm
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            original_frame = frame.copy()
            ball_positions_cm = []
            resized = cv2.resize(frame, (416, 416))
            predictions = model.predict(resized, confidence=30, overlap=20).json()
            scale_x = frame.shape[1] / 416
            scale_y = frame.shape[0] / 416
            for pred in predictions.get('predictions', []):
                x = int(pred['x'] * scale_x)
                y = int(pred['y'] * scale_y)
                w = int(pred['width'] * scale_x)
                h = int(pred['height'] * scale_y)
                label = pred['class']
                color = class_colors.setdefault(label, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                cv2.rectangle(original_frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
                cv2.putText(original_frame, f"{label}", (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cm_coords = pixel_to_cm(x, y)
                if cm_coords is not None:
                    cx, cy = cm_coords
                    # Ignore balls that fall within the ignored area
                    if (ignored_area['x_min'] <= cx <= ignored_area['x_max'] and
                        ignored_area['y_min'] <= cy <= ignored_area['y_max']):
                        continue  # Skip this ball
                    ball_positions_cm.append((cx, cy, label))
            frame_with_grid = draw_metric_grid(original_frame)
            frame_with_route = draw_full_route(frame_with_grid, ball_positions_cm)
            if not output_queue.full():
                output_queue.put(frame_with_route)

def display_frames():
    global placing_goal, selected_goal
    cv2.namedWindow("Live Object Detection")
    cv2.setMouseCallback("Live Object Detection", click_to_set_corners)
    while True:
        if not output_queue.empty():
            frame = output_queue.get()
            cv2.imshow("Live Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                selected_goal = 'A'
                print("✅ Selected Goal A")
            elif key == ord('2'):
                selected_goal = 'B'
                print("✅ Selected Goal B")

cap_thread = threading.Thread(target=capture_frames, daemon=True)
proc_thread = threading.Thread(target=process_frames, daemon=True)
disp_thread = threading.Thread(target=display_frames, daemon=True)

cap_thread.start()
proc_thread.start()
disp_thread.start()

cap_thread.join()
proc_thread.join()
disp_thread.join()

cap.release()
cv2.destroyAllWindows()
