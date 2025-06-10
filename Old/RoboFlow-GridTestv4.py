from roboflow import Roboflow
import cv2
import random
import threading
import time
import numpy as np
from queue import Queue

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

# ✅ For metric grid calibration
calibration_points = []
homography_matrix = None
real_width_cm = 180  # Actual field width
real_height_cm = 120  # Actual field height
grid_spacing_cm = 5  # Smaller grid squares

# ✅ Mouse click for calibration
def click_to_set_corners(event, x, y, flags, param):
    global calibration_points, homography_matrix
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

# ✅ Grid drawing using homography
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
    return overlay

# ✅ Convert pixel position to cm
def pixel_to_cm(px, py):
    if homography_matrix is None:
        return None
    pt = np.array([[[px, py]]], dtype="float32")
    inv_h = np.linalg.inv(homography_matrix)
    real_pt = cv2.perspectiveTransform(pt, inv_h)[0][0]
    return real_pt

# ✅ Convert cm to grid cell
def cm_to_grid_coords(x_cm, y_cm):
    return int(x_cm // grid_spacing_cm), int(y_cm // grid_spacing_cm)

# ✅ Draw path along the grid from (0,0) to ball
def draw_path_to_ball(frame, x_cm, y_cm):
    if homography_matrix is None:
        return frame
    start = (0, 0)
    end = cm_to_grid_coords(x_cm, y_cm)
    overlay = frame.copy()
    path_color = (0, 255, 255)
    x, y = start
    while x != end[0] or y != end[1]:
        if x < end[0]: x += 1
        elif x > end[0]: x -= 1
        if y < end[1]: y += 1
        elif y > end[1]: y -= 1
        pt_cm = np.array([[[x * grid_spacing_cm, y * grid_spacing_cm]]], dtype='float32')
        pt_px = cv2.perspectiveTransform(pt_cm, homography_matrix)[0][0]
        cv2.circle(overlay, tuple(pt_px.astype(int)), 2, path_color, -1)
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
