import cv2
import random
import threading
import numpy as np
import heapq
import itertools
import socket
import json
from queue import Queue, Empty
from roboflow import Roboflow

# === CONFIGURABLE CONSTANTS ===

# 1) Roboflow credentials and version
ROBOFLOW_API_KEY = "7kMjalIwU9TqGmKM0g4i"
WORKSPACE_NAME   = "pingpong-fafrv"
PROJECT_NAME     = "newpingpongdetector"
VERSION          = 1   # ← use same version for both ball and obstacle detection

# 2) Video capture settings
CAMERA_INDEX     = 1
FRAME_WIDTH      = 1280
FRAME_HEIGHT     = 720
FRAMES_PER_SEC   = 30
BUFFER_SIZE      = 1  # cv2.CAP_PROP_BUFFERSIZE

# 3) Table / grid dimensions (in cm)
REAL_WIDTH_CM    = 180
REAL_HEIGHT_CM   = 120
GRID_SPACING_CM  = 2

# 4) Start & goal definitions (in cm)
START_POINT_CM = (20, 20)
GOAL_A_CM      = (REAL_WIDTH_CM, REAL_HEIGHT_CM // 2)  # (180, 60)
GOAL_B_CM      = None  # Define if needed

GOAL_RANGE = {
    'A': [GOAL_A_CM],
    'B': GOAL_B_CM
}

# 4.1) Obstacle‐buffer (in cm). A* will treat cells within this many cm of any “raw” obstacle as blocked.
OBSTACLE_BUFFER_CM = 10    # ← adjust to increase/decrease buffer

# 4.2) (Derived) how many grid‐cells that buffer corresponds to:
BUFFER_CELLS = int(np.ceil(OBSTACLE_BUFFER_CM / GRID_SPACING_CM))

# 5) How many balls we want to route through (max). Change to 3, 5, etc.
MAX_BALLS_TO_COLLECT = 4   # ← adjust as needed

# 6) Region to ignore (in cm)
IGNORED_AREA = {
    'x_min': 50,
    'x_max': 100,
    'y_min': 50,
    'y_max': 100
}

# 7) Detection thresholds
CONFIDENCE_THRESHOLD = 0.50    # 50%
OVERLAP_THRESHOLD    = 0.05    # 5%
MIN_RED_AREA_PX      = 500     # Minimum pixel area for red‐cross contour
MAX_RED_AREA_CM2     = 400     # Maximum approximate area (cm²) for red contour

# 8) Frame skipping (process every Nth frame)
SKIP_FRAMES = 3

# 9) Robot communication parameters
ROBOT_IP      = "10.225.58.57"
ROBOT_PORT    = 12345
ROBOT_HEADING = "N"

# 10) Visualization & drawing parameters
OBSTACLE_DRAW_RADIUS_PX = 6
GRID_LINE_COLOR         = (100, 100, 100)
PATH_COLOR              = (0, 255, 255)
TEXT_COLOR              = (0, 255, 255)

# 11) Color map seed (for reproducible random colors)
RANDOM_SEED = 42

# === END CONFIGURABLE CONSTANTS ===

# === Roboflow Model Initialization ===
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)

# We use two model objects (same version) for clarity:
model_v1 = project.version(VERSION).model  # for ball detection
model_v3 = project.version(VERSION).model  # for everything else

# === Video Capture Setup ===
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
cap.set(cv2.CAP_PROP_FPS, FRAMES_PER_SEC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# === Threading & Queues ===
frame_queue  = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
stop_event   = threading.Event()

# === Calibration & Homography Globals ===
calibration_points    = []
homography_matrix     = None
inv_homography_matrix = None
grid_overlay          = None

# === Obstacles & Caching ===
obstacles               = set()   # set of (gx, gy) cells
ball_positions_cm       = []      # will store 5-tuples: (cx_cm, cy_cm, lbl, cx_px, cy_px)
last_ball_positions_cm  = []
cached_route            = None
last_selected_goal      = None
pending_route           = None
full_grid_path          = []

# === Color Mapping for Classes ===
random.seed(RANDOM_SEED)
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
    pt       = np.array([[[px, py]]], dtype="float32")
    real_pt  = cv2.perspectiveTransform(pt, inv_homography_matrix)[0][0]
    return real_pt[0], real_pt[1]

def cm_to_grid_coords(x_cm, y_cm):
    return int(x_cm // GRID_SPACING_CM), int(y_cm // GRID_SPACING_CM)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, grid_w, grid_h, obstacles_set):
    """
    A* on 4-neighborhood (no diagonals). 
    start/goal are grid-cell tuples, e.g. (gx, gy).
    Returns a list of grid-cell coords, or [] if no path.
    """
    open_set  = [(0, start)]
    came_from = {}
    g_score   = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # Reconstruct path
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
    """
    Determine if the chosen balls have changed significantly.
    Both lists are lists of 5-tuples: (x_cm, y_cm, lbl, cx_px, cy_px).
    """
    if len(ball_positions) != len(last_positions):
        return True
    for (x, y, lbl, _, _), (lx, ly, llbl, _, _) in zip(ball_positions, last_positions):
        if lbl != llbl or abs(x - lx) > tol_cm or abs(y - ly) > tol_cm:
            return True
    return False

def pick_top_n(balls, start_cm, n=MAX_BALLS_TO_COLLECT):
    """
    Given a list of (x_cm, y_cm, label, cx_px, cy_px),
    pick the n closest to start_cm by Manhattan distance on the grid.
    Returns up to n of those same 5-tuples.
    """
    sx_g, sy_g = cm_to_grid_coords(*start_cm)

    def manh_dist(ball):
        bx_cm, by_cm = ball[0], ball[1]
        bx_g, by_g = cm_to_grid_coords(bx_cm, by_cm)
        return abs(sx_g - bx_g) + abs(sy_g - by_g)

    sorted_balls = sorted(balls, key=manh_dist)
    return sorted_balls[:n]

def get_expanded_obstacles(raw_obstacles):
    """
    Given a set of raw obstacle cells (each as (gx, gy)),
    return a new set that also includes any cell within BUFFER_CELLS (Chebyshev)
    of each raw obstacle.
    """
    expanded = set()
    for (gx, gy) in raw_obstacles:
        for dx in range(-BUFFER_CELLS, BUFFER_CELLS + 1):
            for dy in range(-BUFFER_CELLS, BUFFER_CELLS + 1):
                nx, ny = gx + dx, gy + dy
                max_gx = REAL_WIDTH_CM // GRID_SPACING_CM
                max_gy = REAL_HEIGHT_CM // GRID_SPACING_CM
                if 0 <= nx <= max_gx and 0 <= ny <= max_gy:
                    expanded.add((nx, ny))
    return expanded

def compute_best_route(balls_list, goal_name):
    """
    balls_list: list of 5-tuples (x_cm, y_cm, label, cx_px, cy_px)
    Perform a brute-force TSP: START → each ball → GOAL.
    Returns:
      - route_cm: list of (x_cm, y_cm) in order
      - full_grid_path: flattened list of grid-cell coords via A*
    """
    if len(balls_list) == 0:
        return [], []

    grid_w = REAL_WIDTH_CM // GRID_SPACING_CM
    grid_h = REAL_HEIGHT_CM // GRID_SPACING_CM

    start_cell = cm_to_grid_coords(*START_POINT_CM)
    ball_cells = [cm_to_grid_coords(b[0], b[1]) for b in balls_list]

    goal_point_cm = GOAL_RANGE[goal_name][0]
    goal_cell = cm_to_grid_coords(goal_point_cm[0], goal_point_cm[1])

    # Build [start_cell] + ball_cells
    points = [start_cell] + ball_cells
    n = len(points)
    raw_obs = obstacles
    expanded_obs = get_expanded_obstacles(raw_obs)

    # Precompute pairwise costs & paths among start & balls
    distance_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = points[i], points[j]
            path_ij = astar(pi, pj, grid_w, grid_h, expanded_obs)
            cost_ij = len(path_ij)
            distance_map[(i, j)] = (cost_ij, path_ij)
            distance_map[(j, i)] = (cost_ij, list(reversed(path_ij)))

    # Precompute ball_i → goal
    ball_to_goal_map = {}
    for idx, bcell in enumerate(ball_cells, start=1):
        path_bg = astar(bcell, goal_cell, grid_w, grid_h, expanded_obs)
        cost_bg = len(path_bg)
        ball_to_goal_map[idx] = (cost_bg, path_bg)

    best_sequence = None
    best_cost = None
    best_segments = None

    ball_indices = list(range(1, len(ball_cells) + 1))
    for perm in itertools.permutations(ball_indices):
        total_cost = 0
        segments = []

        # start → first ball
        c01, p01 = distance_map[(0, perm[0])]
        total_cost += c01
        segments.append(((0, perm[0]), p01))

        # between balls
        for k in range(len(perm) - 1):
            i_idx, j_idx = perm[k], perm[k + 1]
            c_ij, p_ij = distance_map[(i_idx, j_idx)]
            total_cost += c_ij
            segments.append(((i_idx, j_idx), p_ij))

        # last ball → goal
        c_last, p_last = ball_to_goal_map[perm[-1]]
        total_cost += c_last
        segments.append(((perm[-1], 'G'), p_last))

        if best_cost is None or total_cost < best_cost:
            best_cost = total_cost
            best_sequence = perm
            best_segments = list(segments)

    # Reconstruct route_cm: START → chosen balls → GOAL
    route_cm = [START_POINT_CM]
    for bi in best_sequence:
        bx_cm, by_cm, _ = balls_list[bi - 1][0:3]
        route_cm.append((bx_cm, by_cm))
    route_cm.append(goal_point_cm)

    # Flatten grid‐cells
    full_path_cells = []
    for (_, _), segment_cells in best_segments:
        full_path_cells.extend(segment_cells)

    return route_cm, full_path_cells

def create_and_cache_grid_overlay():
    global homography_matrix, grid_overlay
    # Build a flat “real-world” canvas showing the 2 cm-spaced grid
    canvas = np.zeros((REAL_HEIGHT_CM + 1, REAL_WIDTH_CM + 1, 3), dtype=np.uint8)
    for x_cm in range(0, REAL_WIDTH_CM + 1, GRID_SPACING_CM):
        cv2.line(canvas, (x_cm, 0), (x_cm, REAL_HEIGHT_CM), GRID_LINE_COLOR, 1)
    for y_cm in range(0, REAL_HEIGHT_CM + 1, GRID_SPACING_CM):
        cv2.line(canvas, (0, y_cm), (REAL_WIDTH_CM, y_cm), GRID_LINE_COLOR, 1)

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
    blended = cv2.addWeighted(frame, 1.0, grid_overlay, 0.5, 0)
    for (gx, gy) in obstacles:
        x_cm, y_cm = gx * GRID_SPACING_CM, gy * GRID_SPACING_CM
        pt_cm = np.array([[[x_cm, y_cm]]], dtype="float32")
        pt_px = cv2.perspectiveTransform(pt_cm, homography_matrix)[0][0]
        cv2.circle(blended, (int(pt_px[0]), int(pt_px[1])), OBSTACLE_DRAW_RADIUS_PX, (0, 0, 255), -1)
    return blended

def draw_full_route(frame, ball_positions):
    global cached_route, last_ball_positions_cm, last_selected_goal, pending_route, full_grid_path

    if homography_matrix is None:
        return frame

    # 1) Pick up to MAX_BALLS_TO_COLLECT balls (closest by Manhattan on grid)
    chosen_balls = pick_top_n(ball_positions, START_POINT_CM, n=MAX_BALLS_TO_COLLECT)

    # --- Highlight each chosen ball in pixel‐space ---
    for (cx_cm, cy_cm, lbl, cx_px, cy_px) in chosen_balls:
        # Draw a green circle around that ball’s pixel center
        cv2.circle(frame, (int(cx_px), int(cy_px)), 10, (0, 255, 0), 3)
        cv2.putText(frame, lbl, (int(cx_px) - 10, int(cy_px) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 2) Decide if route needs updating
    route_changed = (
        cached_route is None
        or significant_change(chosen_balls, last_ball_positions_cm)
        or (last_selected_goal != selected_goal)
    )

    if route_changed:
        if len(ball_positions) > MAX_BALLS_TO_COLLECT:
            coords = [(round(b[0], 1), round(b[1], 1), b[2]) for b in chosen_balls]
            print(f"📋 Using only these {MAX_BALLS_TO_COLLECT} balls: {coords}")

        route_cm, grid_cells = compute_best_route(chosen_balls, selected_goal)
        cached_route            = route_cm
        last_ball_positions_cm  = chosen_balls.copy()
        last_selected_goal      = selected_goal
        full_grid_path          = grid_cells
    else:
        route_cm = cached_route

    pending_route = route_cm

    # 3) Draw the A* route on top of the frame
    overlay = frame.copy()
    total_cm = 0

    grid_w = REAL_WIDTH_CM // GRID_SPACING_CM
    grid_h = REAL_HEIGHT_CM // GRID_SPACING_CM

    for i in range(len(route_cm) - 1):
        start_cell = cm_to_grid_coords(*route_cm[i])
        end_cell   = cm_to_grid_coords(*route_cm[i + 1])
        expanded_obs = get_expanded_obstacles(obstacles)
        path_cells = astar(start_cell, end_cell, grid_w, grid_h, expanded_obs)
        if not path_cells:
            continue

        for j in range(len(path_cells)):
            prev_cell = path_cells[j - 1] if j > 0 else path_cells[0]
            curr_cell = path_cells[j]

            gx0, gy0 = prev_cell
            gx1, gy1 = curr_cell

            x0_cm, y0_cm = gx0 * GRID_SPACING_CM, gy0 * GRID_SPACING_CM
            x1_cm, y1_cm = gx1 * GRID_SPACING_CM, gy1 * GRID_SPACING_CM

            pt0_cm = np.array([[[x0_cm, y0_cm]]], dtype="float32")
            pt1_cm = np.array([[[x1_cm, y1_cm]]], dtype="float32")
            pt0_px = cv2.perspectiveTransform(pt0_cm, homography_matrix)[0][0]
            pt1_px = cv2.perspectiveTransform(pt1_cm, homography_matrix)[0][0]

            cv2.line(
                overlay,
                (int(pt0_px[0]), int(pt0_px[1])),
                (int(pt1_px[0]), int(pt1_px[1])),
                PATH_COLOR,
                3
            )
            total_cm += GRID_SPACING_CM

    cv2.putText(
        overlay,
        f"Total Path: {total_cm}cm to Goal {selected_goal}",
        (10, overlay.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
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
                [REAL_WIDTH_CM, 0],
                [REAL_WIDTH_CM, REAL_HEIGHT_CM],
                [0, REAL_HEIGHT_CM]
            ], dtype="float32")
            src_points = np.array(calibration_points, dtype="float32")
            homography_matrix    = cv2.getPerspectiveTransform(dst_points, src_points)
            inv_homography_matrix = np.linalg.inv(homography_matrix)
            print("✅ Homography calculated.")
            create_and_cache_grid_overlay()

    elif event == cv2.EVENT_RBUTTONDOWN and homography_matrix is not None:
        pt_cm = pixel_to_cm(x, y)
        if pt_cm is not None:
            gx, gy = cm_to_grid_coords(pt_cm[0], pt_cm[1])
            max_gx = REAL_WIDTH_CM // GRID_SPACING_CM
            max_gy = REAL_HEIGHT_CM // GRID_SPACING_CM
            if 0 <= gx <= max_gx and 0 <= gy <= max_gy:
                if (gx, gy) in obstacles:
                    obstacles.remove((gx, gy))
                    print(f"🟢 Cleared obstacle at: ({gx}, {gy})")
                else:
                    obstacles.add((gx, gy))
                    print(f"🚧 Added obstacle at: ({gx}, {gy})")

def ensure_outer_edges_walkable():
    max_x = REAL_WIDTH_CM // GRID_SPACING_CM
    max_y = REAL_HEIGHT_CM // GRID_SPACING_CM
    for gx in range(max_x + 1):
        obstacles.discard((gx, 0))
        obstacles.discard((gx, max_y))
    for gy in range(max_y + 1):
        obstacles.discard((0, gy))
        obstacles.discard((max_x, gy))
    print("✅ Outer edges cleared.")

def send_path(ip: str, port: int, grid_path: list, heading: str):
    """
    Opens a TCP socket to (ip, port), serializes the grid_path + heading as JSON,
    and sends it. Expects grid_path as a list of (gx, gy) tuples.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            sock.connect((ip, port))

            payload = {
                "heading": heading,
                "path": [[int(gx), int(gy)] for (gx, gy) in grid_path]
            }
            data = json.dumps(payload).encode("utf-8")
            length_prefix = len(data).to_bytes(4, byteorder='big')
            sock.sendall(length_prefix + data)
            print(f"📨 Sent path to {ip}:{port} → {len(grid_path)} cells, heading={heading}")
    except Exception as e:
        print(f"❌ Failed to send path: {e}")

# === Thread Functions ===

def capture_frames():
    frame_count_local = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count_local += 1
        if frame_count_local % SKIP_FRAMES == 0:
            try:
                frame_queue.put(frame, timeout=0.02)
            except:
                pass
    print("📷 capture_frames exiting")

def process_frames():
    global ball_positions_cm, obstacles

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.02)
        except Empty:
            continue

        original = frame.copy()
        small    = cv2.resize(frame, (416, 416))
        h_full, w_full = original.shape[:2]
        scale_x = w_full / 416
        scale_y = h_full / 416

        # --- 1) DETECT BALLS (using model_v1) ---
        preds_v1 = model_v1.predict(
            small,
            confidence=int(CONFIDENCE_THRESHOLD * 100),
            overlap=int(OVERLAP_THRESHOLD * 100)
        ).json()

        ball_positions_cm.clear()
        for d in preds_v1.get('predictions', []):
            lbl = d['class'].strip().lower()
            if "ball" not in lbl:
                continue

            cx  = int(d['x'] * scale_x)
            cy  = int(d['y'] * scale_y)
            w   = int(d['width'] * scale_x)
            h   = int(d['height'] * scale_y)

            # Color‐code “ball” boxes
            if "white" in lbl:
                color = (200, 200, 255)
            elif "orange" in lbl:
                color = (0, 165, 255)
            else:
                color = class_colors.setdefault(lbl,
                            (random.randint(0,255), random.randint(0,255), random.randint(0,255)))

            x1, y1 = cx - w//2, cy - h//2
            x2, y2 = cx + w//2, cy + h//2
            cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original, lbl, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Convert center‐pixel → real‐world cm coords
            cm_coords = pixel_to_cm(cx, cy)
            if cm_coords is not None:
                cx_cm, cy_cm = cm_coords
                # ignore if inside the “ignored area”
                if not (
                    IGNORED_AREA['x_min'] <= cx_cm <= IGNORED_AREA['x_max']
                    and IGNORED_AREA['y_min'] <= cy_cm <= IGNORED_AREA['y_max']
                ):
                    # Store (cm_x, cm_y, label, px_x, px_y)
                    ball_positions_cm.append((cx_cm, cy_cm, lbl, cx, cy))

        # --- 2) DETECT other objects (using model_v3) and mark them as obstacles ---
        preds_v3 = model_v3.predict(
            small,
            confidence=int(CONFIDENCE_THRESHOLD * 100),
            overlap=int(OVERLAP_THRESHOLD * 100)
        ).json()

        # We do not clear obstacles here; we merge new detections each frame,
        # plus red‐cross obstacles below, plus any manual clicks.
        for d in preds_v3.get('predictions', []):
            lbl = d['class'].strip().lower()
            if "ball" in lbl:
                continue  # skip balls here—already handled

            cx  = int(d['x'] * scale_x)
            cy  = int(d['y'] * scale_y)
            w   = int(d['width'] * scale_x)
            h   = int(d['height'] * scale_y)

            color = class_colors.setdefault(lbl, (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            x1, y1 = cx - w//2, cy - h//2
            x2, y2 = cx + w//2, cy + h//2
            cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original, lbl, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Sample inside bounding box every 10px → convert to cm → grid → add to obstacles
            if homography_matrix is not None:
                max_gx = REAL_WIDTH_CM // GRID_SPACING_CM
                max_gy = REAL_HEIGHT_CM // GRID_SPACING_CM
                for sx in range(x1, x2, 10):
                    for sy in range(y1, y2, 10):
                        real = pixel_to_cm(sx, sy)
                        if real is None:
                            continue
                        gx, gy = cm_to_grid_coords(real[0], real[1])
                        if 0 <= gx <= max_gx and 0 <= gy <= max_gy:
                            obstacles.add((gx, gy))

        # --- 3) DETECT RED CROSS (size‐filtered) and merge those cells into obstacles ---
        if homography_matrix is not None:
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel)

            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_contour = None
            best_contour_area = 0

            px_per_cm_x = w_full / REAL_WIDTH_CM
            px_per_cm_y = h_full / REAL_HEIGHT_CM

            for cnt in contours:
                area_px = cv2.contourArea(cnt)
                if area_px < MIN_RED_AREA_PX:
                    continue

                x_r, y_r, w_r, h_r = cv2.boundingRect(cnt)
                approx_cm_w = w_r / px_per_cm_x
                approx_cm_h = h_r / px_per_cm_y
                approx_cm_area = approx_cm_w * approx_cm_h

                if approx_cm_area > MAX_RED_AREA_CM2:
                    continue

                if area_px > best_contour_area:
                    best_contour = cnt
                    best_contour_area = area_px

            new_obstacles = set()
            if best_contour is not None:
                cv2.drawContours(original, [best_contour], -1, (0, 255, 0), 2)

                bx, by, bw_cnt, bh_cnt = cv2.boundingRect(best_contour)
                max_gx = REAL_WIDTH_CM // GRID_SPACING_CM
                max_gy = REAL_HEIGHT_CM // GRID_SPACING_CM

                for sx in range(bx, bx + bw_cnt, 10):
                    for sy in range(by, by + bh_cnt, 10):
                        if cv2.pointPolygonTest(best_contour, (sx, sy), False) >= 0:
                            real = pixel_to_cm(sx, sy)
                            if real is None:
                                continue
                            gx, gy = cm_to_grid_coords(real[0], real[1])
                            if 0 <= gx <= max_gx and 0 <= gy <= max_gy:
                                new_obstacles.add((gx, gy))

            obstacles |= new_obstacles  # merge red‐cross obstacles

        # --- 4) DRAW GRID + ROUTE (A*) + HIGHLIGHT BALLS ---
        frame_with_grid  = draw_metric_grid(original)
        frame_with_route = draw_full_route(frame_with_grid, ball_positions_cm)

        try:
            output_queue.put(frame_with_route, timeout=0.02)
        except:
            pass

    print("🖥️ process_frames exiting")

def display_frames():
    global selected_goal, full_grid_path
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
                if full_grid_path:
                    send_path(ROBOT_IP, ROBOT_PORT, full_grid_path, ROBOT_HEADING)
                else:
                    print("⚠️ No full_grid_path available to send.")
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
            if full_grid_path:
                send_path(ROBOT_IP, ROBOT_PORT, full_grid_path, ROBOT_HEADING)
            else:
                print("⚠️ No full_grid_path available to send.")

    print("🖼️ display_frames exiting")

# === Main Thread ===

if __name__ == "__main__":
    # Ensure outer edges are always walkable
    ensure_outer_edges_walkable()

    # Default selected goal
    selected_goal = 'A'

    cap_thread = threading.Thread(target=capture_frames)
    proc_thread = threading.Thread(target=process_frames)
    disp_thread = threading.Thread(target=display_frames)

    cap_thread.start()
    proc_thread.start()
    disp_thread.start()

    disp_thread.join()
    cap_thread.join()
    proc_thread.join()

    cap.release()
    cv2.destroyAllWindows()
    print("✂️ Exiting cleanly")
