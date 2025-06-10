import cv2
import random
import threading
import numpy as np
import heapq
from queue import Queue, Empty
from roboflow import Roboflow
import socket
import json
import itertools

# === Roboflow Model Initialization ===
rf = Roboflow(api_key="7kMjalIwU9TqGmKM0g4i")
project = rf.workspace("pingpong-fafrv").project("pingpongdetector-rqboj")
model = project.version(3).model

# === Video Capture Setup ===
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# === Threading & Queues ===
frame_queue  = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
stop_event   = threading.Event()

# === Calibration & Homography Globals ===
calibration_points    = []
homography_matrix     = None
inv_homography_matrix = None
grid_overlay          = None

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
full_grid_path          = []  # Store the raw grid‐cell path for sending

# === Simple Tracking Buffer (pixel coords) ===
buffered_positions_px = []  # holds last up to 3 (cx, cy)

# === Start & Goal Definitions ===
start_point_cm = (20, 20)
# Now go to the center of the goal at y = 60cm, x at table edge (180cm)
goal_range = {
    'A': [(real_width_cm, 60)],
    'B': None  # Define B if you ever need it
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
    pt       = np.array([[[px, py]]], dtype="float32")
    real_pt  = cv2.perspectiveTransform(pt, inv_homography_matrix)[0][0]
    return real_pt[0], real_pt[1]

def cm_to_grid_coords(x_cm, y_cm):
    return int(x_cm // grid_spacing_cm), int(y_cm // grid_spacing_cm)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, grid_w, grid_h, obstacles_set):
    """
    Simple A* on 4‐neighborhood (no diagonals).
    start/goal are grid‐cell tuples, e.g. (gx, gy).
    Returns a list of grid‐cell coordinates, or [] if no path.
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
    if len(ball_positions) != len(last_positions):
        return True
    for (x, y, lbl), (lx, ly, llbl) in zip(ball_positions, last_positions):
        if lbl != llbl or abs(x - lx) > tol_cm or abs(y - ly) > tol_cm:
            return True
    return False

def pick_top_four(balls, start_cm):
    """
    Given a list of (x_cm, y_cm, label), pick the four closest to `start_cm`
    in terms of Manhattan distance on the grid. Returns up to 4 balls.
    """
    sx_g, sy_g = cm_to_grid_coords(*start_cm)

    def manh_dist(ball):
        bx_g, by_g = cm_to_grid_coords(ball[0], ball[1])
        return abs(sx_g - bx_g) + abs(sy_g - by_g)

    sorted_balls = sorted(balls, key=manh_dist)
    return sorted_balls[:4]

def compute_best_route_for_four(four_balls, goal_name):
    """
    Given exactly up to four balls (list of (x_cm, y_cm, label)) and a goal name ('A'),
    perform a small‐scale TSP:
      start_point_cm → (ball1 → ball2 → ball3 → ball4 in some order) → fixed goal (180, 60).
    Returns both:
      - route_cm: a list of (x_cm, y_cm) points in the chosen order (start, each ball, goal)
      - full_grid_path: the combined grid‐cell path (list of (gx, gy) tuples) from A* along each segment
    """
    grid_w = real_width_cm // grid_spacing_cm
    grid_h = real_height_cm // grid_spacing_cm

    # Convert start to grid cell:
    start_cell = cm_to_grid_coords(*start_point_cm)
    # Convert each ball center to grid cell:
    ball_cells = [cm_to_grid_coords(ball[0], ball[1]) for ball in four_balls]

    # Fixed goal center: (180,60) → grid cell:
    goal_point_cm = goal_range[goal_name][0]  # (180,60)
    goal_cell = cm_to_grid_coords(goal_point_cm[0], goal_point_cm[1])

    # Build point list for pairwise distances: [start] + ball_cells
    points = [start_cell] + ball_cells
    n = len(points)  # 1 + up to 4

    # Precompute A* distances & paths between all pairs in points
    distance_map = {}
    for i in range(n):
        for j in range(i+1, n):
            pi = points[i]
            pj = points[j]
            path = astar(pi, pj, grid_w, grid_h, obstacles)
            cost = len(path)
            distance_map[(i, j)] = (cost, path)
            distance_map[(j, i)] = (cost, list(reversed(path)))

    # Precompute ball_i → goal distance & path
    ball_to_goal_map = {}
    for bi, bcell in enumerate(ball_cells, start=1):
        path_bg = astar(bcell, goal_cell, grid_w, grid_h, obstacles)
        cost_bg = len(path_bg)
        ball_to_goal_map[bi] = (cost_bg, path_bg)

    # Now brute‐force permutations of {1..len(ball_cells)}
    best_sequence = None
    best_total_cost = None
    best_segments = None

    ball_indices = list(range(1, len(ball_cells) + 1))  # [1,2,3,4]
    for perm in itertools.permutations(ball_indices):
        total_cost = 0
        segments = []
        # start → first ball
        c01, p01 = distance_map[(0, perm[0])]
        total_cost += c01
        segments.append(((0, perm[0]), p01))
        # between balls
        for k in range(len(perm)-1):
            i = perm[k]
            j = perm[k+1]
            c, p = distance_map[(i, j)]
            total_cost += c
            segments.append(((i, j), p))
        # last ball → goal
        c_bg, p_bg = ball_to_goal_map[perm[-1]]
        total_cost += c_bg
        segments.append(((perm[-1], 'G'), p_bg))
        if best_total_cost is None or total_cost < best_total_cost:
            best_total_cost = total_cost
            best_sequence = perm
            best_segments = list(segments)

    if best_sequence is None:
        return [], []

    # Reconstruct route_cm: start_cm → four ball centers → goal_cm
    route_cm = [start_point_cm]
    for bi in best_sequence:
        bx_cm, by_cm, _ = four_balls[bi-1]
        route_cm.append((bx_cm, by_cm))
    route_cm.append(goal_point_cm)

    # Flatten grid paths
    full_path_cells = []
    for (_, _), path_cells in best_segments:
        full_path_cells.extend(path_cells)

    return route_cm, full_path_cells

def create_and_cache_grid_overlay():
    global homography_matrix, grid_overlay
    # Build a flat “real‐world” canvas showing the 2 cm‐spaced grid
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
    blended = cv2.addWeighted(frame, 1.0, grid_overlay, 0.5, 0)
    for (gx, gy) in obstacles:
        x_cm, y_cm = gx * grid_spacing_cm, gy * grid_spacing_cm
        pt_cm = np.array([[[x_cm, y_cm]]], dtype="float32")
        pt_px = cv2.perspectiveTransform(pt_cm, homography_matrix)[0][0]
        cv2.circle(blended, (int(pt_px[0]), int(pt_px[1])), 6, (0, 0, 255), -1)
    return blended

def draw_full_route(frame, ball_positions):
    global cached_route, last_ball_positions_cm, last_selected_goal, pending_route, selected_goal, full_grid_path

    if homography_matrix is None:
        return frame

    # 1) Pick exactly four balls (closest by Manhattan on grid)
    four_balls = pick_top_four(ball_positions, start_point_cm)

    # 2) Decide if route needs updating
    route_changed = (
        cached_route is None
        or significant_change(four_balls, last_ball_positions_cm)
        or selected_goal != last_selected_goal
    )

    if route_changed:
        if len(ball_positions) > 4:
            coords = [(round(b[0], 1), round(b[1], 1), b[2]) for b in four_balls]
            print(f"📋 Using only these four balls: {coords}")

        # Compute the best‐TSP route through those four plus fixed goal
        route_cm, grid_cells = compute_best_route_for_four(four_balls, selected_goal)
        cached_route = route_cm
        last_ball_positions_cm = four_balls.copy()
        last_selected_goal = selected_goal
        full_grid_path = grid_cells
    else:
        route_cm = cached_route

    pending_route = route_cm

    # Draw that route on top of the frame
    overlay = frame.copy()
    path_color = (0, 255, 255)
    total_cm = 0

    grid_w = real_width_cm // grid_spacing_cm
    grid_h = real_height_cm // grid_spacing_cm

    for i in range(len(route_cm) - 1):
        start_cell = cm_to_grid_coords(*route_cm[i])
        end_cell   = cm_to_grid_coords(*route_cm[i + 1])
        path_cells = astar(start_cell, end_cell, grid_w, grid_h, obstacles)
        if not path_cells:
            continue

        for j in range(len(path_cells)):
            prev_cell = path_cells[j - 1] if j > 0 else path_cells[0]
            curr_cell = path_cells[j]

            gx0, gy0 = prev_cell
            gx1, gy1 = curr_cell

            x0_cm, y0_cm = gx0 * grid_spacing_cm, gy0 * grid_spacing_cm
            x1_cm, y1_cm = gx1 * grid_spacing_cm, gy1 * grid_spacing_cm

            pt0_cm = np.array([[[x0_cm, y0_cm]]], dtype="float32")
            pt1_cm = np.array([[[x1_cm, y1_cm]]], dtype="float32")
            pt0_px = cv2.perspectiveTransform(pt0_cm, homography_matrix)[0][0]
            pt1_px = cv2.perspectiveTransform(pt1_cm, homography_matrix)[0][0]

            cv2.line(
                overlay,
                (int(pt0_px[0]), int(pt0_px[1])),
                (int(pt1_px[0]), int(pt1_px[1])),
                path_color,
                3
            )
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
            homography_matrix    = cv2.getPerspectiveTransform(dst_points, src_points)
            inv_homography_matrix = np.linalg.inv(homography_matrix)
            print("✅ Homography calculated.")
            create_and_cache_grid_overlay()

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
                "path": [ [int(gx), int(gy)] for (gx, gy) in grid_path ]
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
        if frame_count_local % skip_frames == 0:
            try:
                frame_queue.put(frame, timeout=0.02)
            except:
                pass
    print("📷 capture_frames exiting")

def process_frames():
    global ball_positions_cm, buffered_positions_px, obstacles

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.02)
        except Empty:
            continue

        original = frame.copy()

        # --- 1) DEEP-LEARNING INFERENCE AT 416×416 (Baseline) ---
        small = cv2.resize(frame, (416, 416))
        preds = model.predict(small, confidence=30, overlap=20).json()

        h_full, w_full = original.shape[:2]
        scale_x = w_full / 416
        scale_y = h_full / 416

        # Collect raw pixel centers
        raw_detections = []
        for d in preds.get('predictions', []):
            cx  = int(d['x'] * scale_x)
            cy  = int(d['y'] * scale_y)
            w   = int(d['width'] * scale_x)
            h   = int(d['height'] * scale_y)
            lbl = d['class']
            ll  = lbl.strip().lower()

            if "white" in ll:
                color = (200, 200, 255)
            elif "orange" in ll:
                color = (0, 165, 255)
            else:
                color = class_colors.setdefault(
                    lbl,
                    (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                )

            x1, y1 = cx - w//2, cy - h//2
            x2, y2 = cx + w//2, cy + h//2
            cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)
            cv2.putText(original, lbl, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            raw_detections.append((cx, cy, lbl))

        # --- 2) SIMPLE PIXEL-LEVEL TRACKING (AVERAGE OVER LAST 3) ---
        if raw_detections:
            # pick the first detection (or you could pick the one closest to last)
            bx, by, blbl = raw_detections[0]
            buffered_positions_px.append((bx, by))
            if len(buffered_positions_px) > 3:
                buffered_positions_px.pop(0)
            avg_px = sum(p[0] for p in buffered_positions_px) / len(buffered_positions_px)
            avg_py = sum(p[1] for p in buffered_positions_px) / len(buffered_positions_px)
            cv2.circle(original, (int(avg_px), int(avg_py)), 5, (0,255,255), -1)
            # Project averaged pixel->cm
            cm_coords = pixel_to_cm(avg_px, avg_py)
            if cm_coords is not None:
                cx_cm, cy_cm = cm_coords
                if not (
                    ignored_area['x_min'] <= cx_cm <= ignored_area['x_max']
                    and ignored_area['y_min'] <= cy_cm <= ignored_area['y_max']
                ):
                    ball_positions_cm = [(cx_cm, cy_cm, blbl)]
                else:
                    ball_positions_cm = []
            else:
                ball_positions_cm = []
        else:
            buffered_positions_px.clear()
            ball_positions_cm = []

        # --- 3) DETECT RED CROSS (WITH SIZE FILTER) FOR OBSTACLES ---
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

            px_per_cm_x = w_full / real_width_cm
            px_per_cm_y = h_full / real_height_cm

            for cnt in contours:
                area_px = cv2.contourArea(cnt)
                if area_px < 500:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                approx_cm_w = bw / px_per_cm_x
                approx_cm_h = bh / px_per_cm_y
                approx_cm_area = approx_cm_w * approx_cm_h

                if approx_cm_area > 400:
                    continue

                if area_px > best_contour_area:
                    best_contour = cnt
                    best_contour_area = area_px

            new_obstacles = set()
            if best_contour is not None:
                cv2.drawContours(original, [best_contour], -1, (0, 255, 0), 2)

                bx, by, bw, bh = cv2.boundingRect(best_contour)
                for sx in range(bx, bx + bw, 10):
                    for sy in range(by, by + bh, 10):
                        if cv2.pointPolygonTest(best_contour, (sx, sy), False) >= 0:
                            real = pixel_to_cm(sx, sy)
                            if real is None:
                                continue
                            gx, gy = cm_to_grid_coords(real[0], real[1])
                            max_gx = real_width_cm // grid_spacing_cm
                            max_gy = real_height_cm // grid_spacing_cm
                            if 0 <= gx <= max_gx and 0 <= gy <= max_gy:
                                new_obstacles.add((gx, gy))

            obstacles |= new_obstacles  # merge instead of overwrite

        # --- 4) DRAW GRID + ROUTE (A*) ---
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
                    robot_ip = "10.225.58.57"
                    robot_port = 12345
                    heading = "N"
                    send_path(robot_ip, robot_port, full_grid_path, heading)
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
                robot_ip = "10.225.58.57"
                robot_port = 12345
                heading = "N"
                send_path(robot_ip, robot_port, full_grid_path, heading)
            else:
                print("⚠️ No full_grid_path available to send.")

    print("🖼️ display_frames exiting")

# === Main Thread ===

if __name__ == "__main__":
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
