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

# ✅ Mouse click for calibration and obstacle marking
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
    elif event == cv2.EVENT_RBUTTONDOWN and homography_matrix is not None:
        pt_cm = pixel_to_cm(x, y)
        if pt_cm is not None:
            gx, gy = cm_to_grid_coords(pt_cm[0], pt_cm[1])
            if 0 <= gx <= real_width_cm // grid_spacing_cm and 0 <= gy <= real_height_cm // grid_spacing_cm:
                obstacles.symmetric_difference_update({(gx, gy)})
                print(f"🚧 Toggled obstacle at grid cell: ({gx}, {gy})")

# ✅ Grid and path drawing
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
    return overlay

# ✅ Conversion helpers
def pixel_to_cm(px, py):
    if homography_matrix is None:
        return None
    pt = np.array([[[px, py]]], dtype="float32")
    inv_h = np.linalg.inv(homography_matrix)
    real_pt = cv2.perspectiveTransform(pt, inv_h)[0][0]
    return real_pt

def cm_to_grid_coords(x_cm, y_cm):
    return int(x_cm // grid_spacing_cm), int(y_cm // grid_spacing_cm)

def grid_to_cm_coords(gx, gy):
    return gx * grid_spacing_cm, gy * grid_spacing_cm

# ✅ A* pathfinding
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

def draw_path_to_ball(frame, x_cm, y_cm):
    if homography_matrix is None:
        return frame
    end = cm_to_grid_coords(x_cm, y_cm)
    path = astar((0, 0), end)
    if not path:
        return frame
    overlay = frame.copy()
    path_color = (0, 255, 255)
    prev_pt = None
    for gx, gy in path:
        pt_cm = np.array([[[gx * grid_spacing_cm, gy * grid_spacing_cm]]], dtype='float32')
        pt_px = cv2.perspectiveTransform(pt_cm, homography_matrix)[0][0]
        pt_px = tuple(pt_px.astype(int))
        if prev_pt is not None:
            cv2.line(overlay, prev_pt, pt_px, path_color, 3)
        prev_pt = pt_px
    # Optional: show path length
    cv2.putText(overlay, f"Path: {len(path) * grid_spacing_cm}cm", (10, overlay.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, path_color, 2)
    return overlay

# ✅ Frame capture
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

# ✅ Object detection and drawing
def process_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            original_frame = frame.copy()
            start_time = time.time()
            resized = cv2.resize(frame, (416, 416))
            predictions = model.predict(resized, confidence=30, overlap=20).json()
            object_counts = {}
            scale_x = frame.shape[1] / 416
            scale_y = frame.shape[0] / 416
            for pred in predictions.get('predictions', []):
                x = int(pred['x'] * scale_x)
                y = int(pred['y'] * scale_y)
                w = int(pred['width'] * scale_x)
                h = int(pred['height'] * scale_y)
                label = pred['class']
                confidence = pred['confidence']
                object_counts[label] = object_counts.get(label, 0) + 1
                if label not in class_colors:
                    class_colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                color = class_colors[label]
                cv2.rectangle(original_frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
                cv2.putText(original_frame, f"{label}: {confidence:.2f}", (x - w // 2, y - h // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cm_coords = pixel_to_cm(x, y)
                if cm_coords is not None:
                    cx, cy = cm_coords
                    cv2.putText(original_frame, f"{cx:.1f}cm, {cy:.1f}cm", (x + 5, y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    original_frame = draw_path_to_ball(original_frame, cx, cy)
            y_offset = 30
            for obj, count in object_counts.items():
                cv2.putText(original_frame, f"{obj}: {count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30
            frame_with_grid = draw_metric_grid(original_frame)
            if not output_queue.full():
                output_queue.put(frame_with_grid)
            end_time = time.time()
            print(f"Inference Time: {end_time - start_time:.2f} sec | Detected Objects: {object_counts}")

# ✅ Display logic
def display_frames():
    cv2.namedWindow("Live Object Detection")
    cv2.setMouseCallback("Live Object Detection", click_to_set_corners)
    while True:
        if not output_queue.empty():
            frame = output_queue.get()
            cv2.imshow("Live Object Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def mark_center_obstacle():
    if homography_matrix is None:
        return

    cx, cy = real_width_cm / 2, real_height_cm / 2
    half = 10  # 10cm in each direction = 20cm total

    for x_cm in range(int(cx - half), int(cx + half), grid_spacing_cm):
        for y_cm in range(int(cy - half), int(cy + half), grid_spacing_cm):
            gx, gy = cm_to_grid_coords(x_cm, y_cm)
            obstacles.add((gx, gy))

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
