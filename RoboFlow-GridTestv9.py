import cv2
import random
import threading
import numpy as np
import heapq
from queue import Queue, Empty
from roboflow import Roboflow

# === Roboflow Model Initialization ===
rf = Roboflow(api_key="7kMjalIwU9TqGmKM0g4i")
project = rf.workspace("pingpong-fafrv").project("pingpongdetector-rqboj")
model = project.version(2).model

# === Video Capture Setup ===
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# === Threading & Queues ===
frame_queue = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
stop_event = threading.Event()

# === Calibration & Homography Globals ===
calibration_points = []
homography_matrix = None
inv_homography_matrix = None
grid_overlay = None

# === Real‐World Dimensions (cm) ===
real_width_cm   = 180
real_height_cm  = 120
grid_spacing_cm = 2

# === Grid & Obstacles ===
obstacles = set()

# === Ball Positions & Route Caching ===
ball_positions_cm       = []
cached_route            = None
last_ball_positions_cm  = []
last_selected_goal      = None
pending_route           = None

# === Start & Goal Definitions ===
start_point_cm = (20, 20)
goal_range = {
    'A': [(real_width_cm, y) for y in range(56, 65)],
    'B': None  # Define B if needed
}
selected_goal = 'A'

# === Ignored Detection Area (cm) ===
ignored_area = {
    'x_min': 50,
    'x_max': 100,
    'y_min': 50,
    'y_max': 100
}

# === Frame Skipping ===
skip_frames = 3

# === Color Mapping for Classes ===
random.seed(42)
class_colors = {}

# === Utility Functions ===

def save_route_to_file(route_cm, filename="route.txt"):
    try:
        with open(filename, "w") as f:
            for x, y in route_cm:
                f.write(f"{x:.2f},{y:.2f}\n")
        print("📦 Route saved to route.txt")
    except Exception as e:
        print("❌ Error saving route:", e)

def pixel_to_cm(px, py):
    if inv_homography_matrix is None:
        return None
    pt = np.array([[[px, py]]], dtype="float32")
    real_pt = cv2.perspectiveTransform(pt, inv_homography_matrix)[0][0]
    return real_pt[0], real_pt[1]

def cm_to_grid_coords(x_cm, y_cm):
    return int(x_cm // grid_spacing_cm), int(y_cm // grid_spacing_cm)

def heuristic(cell_a, cell_b):
    return abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])

def astar(start, goal, grid_w, grid_h, obstacles_set):
    open_set = [(0, start)]
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
            nx, ny = x + dx, y + dy
            if 0 <= nx <= grid_w and 0 <= ny <= grid_h and (nx, ny) not in obstacles_set:
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, (nx, ny)))
    return []

def significant_change(ball_positions, last_positions, tol_cm=1.0):
    if len(ball_positions) != len(last_positions):
        return True
    for (x, y, lbl), (lx, ly, llbl) in zip(ball_positions, last_positions):
        if lbl != llbl or abs(x - lx) > tol_cm or abs(y - ly) > tol_cm:
            return True
    return False

def compute_route_from_points(ball_positions, goal_name):
    non_orange = [b for b in ball_positions if "orange" not in b[2].lower()]
    orange    = [b for b in ball_positions if "orange" in b[2].lower()]
    route_cm = [start_point_cm]
    current = start_point_cm

    remaining = non_orange[:]
    while remaining:
        def dist_to_current(ball):
            gx_c, gy_c = cm_to_grid_coords(*current)
            gx_b, gy_b = cm_to_grid_coords(ball[0], ball[1])
            return abs(gx_c - gx_b) + abs(gy_c - gy_b)
        nxt = min(remaining, key=dist_to_current)
        route_cm.append((nxt[0], nxt[1]))
        remaining.remove(nxt)
        current = (nxt[0], nxt[1])

    if orange:
        route_cm.append((orange[0][0], orange[0][1]))
        current = (orange[0][0], orange[0][1])

    goals = goal_range.get(goal_name, [])
    if goals:
        def dist_to_goal_pt(pt):
            gx_c, gy_c = cm_to_grid_coords(*current)
            gx_g, gy_g = cm_to_grid_coords(pt[0], pt[1])
            return abs(gx_c - gx_g) + abs(gy_c - gy_g)
        target = min(goals, key=dist_to_goal_pt)
        route_cm.append(target)

    return route_cm

def create_and_cache_grid_overlay():
    global homography_matrix, grid_overlay
    canvas = np.zeros((real_height_cm + 1, real_width_cm + 1, 3), dtype=np.uint8)
    for x_cm in range(0, real_width_cm + 1, grid_spacing_cm):
        cv2.line(canvas, (x_cm, 0), (x_cm, real_height_cm), (100, 100, 100), 1)
    for y_cm in range(0, real_height_cm + 1, grid_spacing_cm):
        cv2.line(canvas, (0, y_cm), (real_width_cm, y_cm), (100, 100, 100), 1)

    height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width_px  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    grid_overlay = cv2.warpPerspective(
        canvas,
        homography_matrix,
        (width_px, height_px),
        flags=cv2.INTER_LINEAR
    )
    print("🗺️ Cached grid overlay.")

def draw_metric_grid(frame):
    if grid_overlay is None:
        return frame
    # Blend the precomputed grid with the current frame
    blended = cv2.addWeighted(frame, 1.0, grid_overlay, 0.5, 0)
    # Draw obstacles on top:
    for (gx, gy) in obstacles:
        # Convert grid cell to cm
        x_cm = gx * grid_spacing_cm
        y_cm = gy * grid_spacing_cm
        pt_cm = np.array([[[x_cm, y_cm]]], dtype="float32")
        pt_px = cv2.perspectiveTransform(pt_cm, homography_matrix)[0][0]
        # Draw a filled red circle to mark obstacle
        cv2.circle(blended, (int(pt_px[0]), int(pt_px[1])), 6, (0, 0, 255), -1)
    return blended

def draw_full_route(frame, ball_positions):
    global cached_route, last_ball_positions_cm, last_selected_goal, pending_route, selected_goal

    # If homography isn't set yet, just return the frame
    if homography_matrix is None:
        return frame

    # Determine if we need to recompute the route
    route_changed = (
        cached_route is None
        or significant_change(ball_positions, last_ball_positions_cm)
        or selected_goal != last_selected_goal
    )

    if route_changed:
        route_cm = compute_route_from_points(ball_positions, selected_goal)
        cached_route = route_cm
        last_ball_positions_cm = ball_positions.copy()
        last_selected_goal = selected_goal
    else:
        route_cm = cached_route

    pending_route = route_cm

    grid_w = real_width_cm // grid_spacing_cm
    grid_h = real_height_cm // grid_spacing_cm

    all_pixel_pts = []
    for i in range(len(route_cm) - 1):
        start_cell = cm_to_grid_coords(*route_cm[i])
        end_cell   = cm_to_grid_coords(*route_cm[i+1])
        path_cells = astar(start_cell, end_cell, grid_w, grid_h, obstacles)
        if not path_cells:
            continue

        # Convert each (gx,gy) to (x_cm,y_cm)
        path_cm = [(gx * grid_spacing_cm, gy * grid_spacing_cm) for gx, gy in path_cells]
        pts_cm = np.array([[[x_cm, y_cm]] for x_cm, y_cm in path_cm], dtype="float32")
        # Batch-transform all points via homography
        pts_px = cv2.perspectiveTransform(pts_cm, homography_matrix)[:, 0, :]

        for (x_px, y_px) in pts_px:
            all_pixel_pts.append((int(x_px), int(y_px)))

    overlay = frame.copy()
    path_color = (0, 255, 255)
    total_cm = 0
    for idx in range(len(all_pixel_pts) - 1):
        x1, y1 = all_pixel_pts[idx]
        x2, y2 = all_pixel_pts[idx + 1]
        cv2.line(overlay, (x1, y1), (x2, y2), path_color, 3)
        total_cm += grid_spacing_cm

    cv2.putText(
        overlay,
        f"Total Path: {total_cm}cm to Goal {selected_goal}",
        (10, overlay.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        path_color,
        2
    )
    return overlay

def click_to_set_corners(event, x, y, flags, param):
    global calibration_points, homography_matrix, inv_homography_matrix, grid_overlay, obstacles
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append([x, y])
            print(f"Corner {len(calibration_points)} set: ({x}, {y})")
        if len(calibration_points) == 4 and homography_matrix is None:
            dst_points = np.array([
                [0, 0],
                [real_width_cm, 0],
                [real_width_cm, real_height_cm],
                [0, real_height_cm]
            ], dtype="float32")
            src_points = np.array(calibration_points, dtype="float32")
            # Note: getPerspectiveTransform(dst, src) maps (real‐cm coords) → (pixel coords)
            homography_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
            inv_homography_matrix = np.linalg.inv(homography_matrix)
            print("✅ Homography calculated.")
            create_and_cache_grid_overlay()
            mark_center_obstacle()
            ensure_outer_edges_walkable()

    elif event == cv2.EVENT_RBUTTONDOWN and homography_matrix is not None:
        pt_cm = pixel_to_cm(x, y)
        if pt_cm is not None:
            gx, gy = cm_to_grid_coords(pt_cm[0], pt_cm[1])
            max_gx = real_width_cm // grid_spacing_cm
            max_gy = real_height_cm // grid_spacing_cm
            if 0 <= gx <= max_gx and 0 <= gy <= max_gy:
                if (gx, gy) in obstacles:
                    obstacles.remove((gx, gy))
                    print(f"🟢 Cleared obstacle at: ({gx}, {gy})")
                else:
                    obstacles.add((gx, gy))
                    print(f"🚧 Added obstacle at: ({gx}, {gy})")

def mark_center_obstacle():
    cx, cy = real_width_cm / 2, real_height_cm / 2
    half = 10
    for x_cm in range(int(cx - half), int(cx + half), grid_spacing_cm):
        for y_cm in range(int(cy - half), int(cy + half), grid_spacing_cm):
            gx, gy = cm_to_grid_coords(x_cm, y_cm)
            obstacles.add((gx, gy))
    print("🟥 Center obstacle (20×20 cm) added.")

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

# === Thread Functions ===

def capture_frames():
    frame_count_local = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count_local += 1
        if frame_count_local % skip_frames == 0:
            try:
                frame_queue.put(frame, timeout=0.02)
            except:
                pass
    print("📷 capture_frames exiting")

def process_frames():
    global ball_positions_cm
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.02)
        except Empty:
            continue

        original = frame.copy()
        small = cv2.resize(frame, (416, 416))
        preds = model.predict(small, confidence=30, overlap=20).json()

        h_full, w_full = original.shape[:2]
        scale_x = w_full / 416
        scale_y = h_full / 416

        ball_positions_cm.clear()
        for d in preds.get('predictions', []):
            cx = int(d['x'] * scale_x)
            cy = int(d['y'] * scale_y)
            w  = int(d['width'] * scale_x)
            h  = int(d['height'] * scale_y)
            lbl = d['class']
            ll = lbl.strip().lower()

            # Hard-coded colors:
            if "white" in ll:
                color = (200, 200, 255)   # very light blue (BGR)
            elif "orange" in ll:
                color = (0, 165, 255)     # orange (BGR)
            else:
                color = class_colors.setdefault(lbl, (random.randint(0,255), random.randint(0,255), random.randint(0,255)))

            x1, y1 = cx - w//2, cy - h//2
            x2, y2 = cx + w//2, cy + h//2
            cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original, lbl, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cm_coords = pixel_to_cm(cx, cy)
            if cm_coords is not None:
                cx_cm, cy_cm = cm_coords
                if not (ignored_area['x_min'] <= cx_cm <= ignored_area['x_max']
                        and ignored_area['y_min'] <= cy_cm <= ignored_area['y_max']):
                    ball_positions_cm.append((cx_cm, cy_cm, lbl))

        frame_with_grid = draw_metric_grid(original)
        frame_with_route = draw_full_route(frame_with_grid, ball_positions_cm)

        try:
            output_queue.put(frame_with_route, timeout=0.02)
        except:
            pass

    print("🖥️ process_frames exiting")

def display_frames():
    global selected_goal
    cv2.namedWindow("Live Object Detection")
    cv2.setMouseCallback("Live Object Detection", click_to_set_corners)

    while not stop_event.is_set():
        try:
            frame = output_queue.get(timeout=0.02)
        except Empty:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break
            elif key == ord('1'):
                selected_goal = 'A'
                print("✅ Selected Goal A")
            elif key == ord('2'):
                selected_goal = 'B'
                print("✅ Selected Goal B")
            elif key == ord('s'):
                if pending_route:
                    save_route_to_file(pending_route)
                else:
                    print("⚠️ No route to save yet.")
            continue

        cv2.imshow("Live Object Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break
        elif key == ord('1'):
            selected_goal = 'A'
            print("✅ Selected Goal A")
        elif key == ord('2'):
            selected_goal = 'B'
            print("✅ Selected Goal B")
        elif key == ord('s'):
            if pending_route:
                save_route_to_file(pending_route)
            else:
                print("⚠️ No route to save yet.")

    print("🖼️ display_frames exiting")

# === Main Thread ===

if __name__ == "__main__":
    cap_thread = threading.Thread(target=capture_frames)
    proc_thread = threading.Thread(target=process_frames)
    disp_thread = threading.Thread(target=display_frames)

    cap_thread.start()
    proc_thread.start()
    disp_thread.start()

    # Wait until the display thread signals exit (via stop_event)
    disp_thread.join()
    cap_thread.join()
    proc_thread.join()

    cap.release()
    cv2.destroyAllWindows()
    print("✂️ Exiting cleanly")
