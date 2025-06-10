from roboflow import Roboflow
import cv2
import random
import threading
import time
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

origin_x, origin_y = None, None
grid_x, grid_y = None, None
origin_lock = threading.Lock()

def click_to_set_grid(event, x, y, flags, param):
    global origin_x, origin_y, grid_x, grid_y
    if event == cv2.EVENT_LBUTTONDOWN:
        with origin_lock:
            if origin_x is None and origin_y is None:
                origin_x, origin_y = x, y
                print(f"First Point (Origin) Set: ({origin_x}, {origin_y})")
            elif grid_x is None and grid_y is None:
                grid_x, grid_y = x, y
                print(f"Second Point (Grid End) Set: ({grid_x}, {grid_y})")
            else:
                origin_x, origin_y, grid_x, grid_y = None, None, None, None
                print("Grid Reset! Click two new points.")

def capture_frames():
    global frame_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames == 0:
            if not frame_queue.full():
                frame_queue.put(frame)  # ✅ Full resolution frame passed forward

def draw_coordinate_system(frame):
    h, w, _ = frame.shape
    with origin_lock:
        if origin_x is None or grid_x is None:
            return frame
        ox, oy = origin_x, origin_y
        gx, gy = grid_x, grid_y

    cv2.rectangle(frame, (ox, oy), (gx, gy), (255, 255, 255), 2)

    step_size = 50
    for x in range(min(ox, gx), max(ox, gx), step_size):
        cv2.line(frame, (x, min(oy, gy)), (x, max(oy, gy)), (50, 50, 50), 1)
    for y in range(min(oy, gy), max(oy, gy), step_size):
        cv2.line(frame, (min(ox, gx), y), (max(ox, gx), y), (50, 50, 50), 1)

    return frame

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

            with origin_lock:
                if origin_x is None or grid_x is None:
                    output_queue.put(original_frame)
                    continue
                ox, oy = origin_x, origin_y

            for pred in predictions.get('predictions', []):
                x = int(pred['x'] * scale_x)
                y = int(pred['y'] * scale_y)
                w = int(pred['width'] * scale_x)
                h = int(pred['height'] * scale_y)
                label = pred['class']
                confidence = pred['confidence']

                relative_x = x - ox
                relative_y = oy - y

                object_counts[label] = object_counts.get(label, 0) + 1

                if label not in class_colors:
                    class_colors[label] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                color = class_colors[label]

                cv2.rectangle(original_frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
                cv2.putText(original_frame, f"{label}: {confidence:.2f}", (x - w // 2, y - h // 2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(original_frame, f"({relative_x}, {relative_y})", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_offset = 30
            for obj, count in object_counts.items():
                cv2.putText(original_frame, f"{obj}: {count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += 30

            frame_with_grid = draw_coordinate_system(original_frame)

            if not output_queue.full():
                output_queue.put(frame_with_grid)

            end_time = time.time()
            print(f"Inference Time: {end_time - start_time:.2f} sec | Detected Objects: {object_counts}")

def display_frames():
    cv2.namedWindow("Live Object Detection")
    cv2.setMouseCallback("Live Object Detection", click_to_set_grid)

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
