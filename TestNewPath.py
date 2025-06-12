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
ROBOFLOW_API_KEY    = "7kMjalIwU9TqGmKM0g4i"
WORKSPACE_NAME      = "pingpong-fafrv"
PROJECT_NAME        = "newpingpongdetector"
VERSION             = 1

CAMERA_INDEX        = 1
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720
FRAMES_PER_SEC      = 30
BUFFER_SIZE         = 1

REAL_WIDTH_CM       = 180
REAL_HEIGHT_CM      = 120
GRID_SPACING_CM     = 2

START_POINT_CM      = (20, 20)
GOAL_A_CM           = (REAL_WIDTH_CM, REAL_HEIGHT_CM // 2)
GOAL_B_CM           = None

GOAL_RANGE = {
    'A': [GOAL_A_CM],
    'B': GOAL_B_CM
}

# obstacle‐expansion buffer (in cm → grid cells)
OBSTACLE_BUFFER_CM  = 10
BUFFER_CELLS        = int(np.ceil(OBSTACLE_BUFFER_CM / GRID_SPACING_CM))

# border‐buffer around the entire field (in cm → grid cells)
BORDER_BUFFER_CM    = 10
BORDER_BUFFER_CELLS = int(np.ceil(BORDER_BUFFER_CM / GRID_SPACING_CM))

MAX_BALLS_TO_COLLECT = 4

IGNORED_AREA = {
    'x_min': 50, 'x_max': 100,
    'y_min': 50, 'y_max': 100
}

CONFIDENCE_THRESHOLD = 0.50
OVERLAP_THRESHOLD    = 0.05
MIN_RED_AREA_PX      = 500
MAX_RED_AREA_CM2     = 400

SKIP_FRAMES         = 3

ROBOT_IP            = "10.137.48.57"
ROBOT_PORT          = 12345
ROBOT_HEADING       = "E"

OBSTACLE_DRAW_RADIUS_PX = 6
GRID_LINE_COLOR         = (100, 100, 100)
PATH_COLOR              = (0, 255, 255)
TEXT_COLOR              = (0, 255, 255)

RANDOM_SEED = 42
TURN_PENALTY = 0.5   # extra cost for changing direction in A*
random.seed(RANDOM_SEED)

# === Roboflow Model Initialization ===
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
model_v1 = project.version(VERSION).model     # for ball detection
model_v3 = project.version(VERSION).model     # for obstacles + robot

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
static_obstacles      = set()    # user‐clicked obstacles
dynamic_obstacles     = set()    # rebuilt each frame
ball_positions_cm     = []
last_ball_positions_cm= []
cached_route          = None
last_selected_goal    = None
pending_route         = None
full_grid_path        = []

# === Color Mapping for Classes ===
class_colors = {}

# === Robot Start Point (dynamic) ===
robot_position_cm = None

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
    return int(x_cm // GRID_SPACING_CM), int(y_cm // GRID_SPACING_CM)

def prune_colinear(path):
    """Drop middle points on straight segments to reduce zig-zags."""
    if len(path) < 3:
        return path
    pruned = [path[0], path[1]]
    for p in path[2:]:
        a, b = pruned[-2], pruned[-1]
        if (b[0]-a[0], b[1]-a[1]) == (p[0]-b[0], p[1]-b[1]):
            pruned[-1] = p
        else:
            pruned.append(p)
    return pruned

def astar(start, goal, grid_w, grid_h, obstacles):
    """
    Turn-penalized A*:
      state is (x, y, incoming_direction)
      cost = 1 for straight, 1 + TURN_PENALTY for turns.
    """
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    open_heap = [(0, 0, start[0], start[1], None)]  # (f, g, x, y, prev_dir)
    came_from = {}
    g_score = {(start[0], start[1], None): 0}

    while open_heap:
        f, g, x, y, prev_dir = heapq.heappop(open_heap)
        if (x, y) == goal:
            # reconstruct path from best goal-state
            end_state = min(
                ((xx,yy,dd) for (xx,yy,dd),gg in g_score.items() if (xx,yy)==goal),
                key=lambda st: g_score[st]
            )
            path = []
            cur = end_state
            while cur in came_from:
                path.append((cur[0], cur[1]))
                cur = came_from[cur]
            path.append(start)
            return path[::-1]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not (0 <= nx <= grid_w and 0 <= ny <= grid_h):
                continue
            if (nx, ny) in obstacles:
                continue

            turn_cost = TURN_PENALTY if (prev_dir and (dx, dy) != prev_dir) else 0
            g2 = g + 1 + turn_cost
            state2 = (nx, ny, (dx, dy))
            if g2 < g_score.get(state2, float('inf')):
                g_score[state2] = g2
                f2 = g2 + abs(nx - goal[0]) + abs(ny - goal[1])
                heapq.heappush(open_heap, (f2, g2, nx, ny, (dx, dy)))
                came_from[state2] = (x, y, prev_dir)

    return []

def get_expanded_obstacles(raw):
    exp = set()
    max_g = REAL_WIDTH_CM // GRID_SPACING_CM
    max_h = REAL_HEIGHT_CM // GRID_SPACING_CM
    for gx,gy in raw:
        for dx in range(-BUFFER_CELLS, BUFFER_CELLS+1):
            for dy in range(-BUFFER_CELLS, BUFFER_CELLS+1):
                nx, ny = gx+dx, gy+dy
                if 0<=nx<=max_g and 0<=ny<=max_h:
                    exp.add((nx,ny))
    return exp

def get_border_buffer_obstacles():
    max_g = REAL_WIDTH_CM // GRID_SPACING_CM
    max_h = REAL_HEIGHT_CM // GRID_SPACING_CM
    border_obs = set()
    # top/bottom bands
    for gx in range(max_g+1):
        for gy in range(BORDER_BUFFER_CELLS):
            border_obs.add((gx, gy))
        for gy in range(max_h - BORDER_BUFFER_CELLS + 1, max_h+1):
            border_obs.add((gx, gy))
    # left/right bands
    for gy in range(max_h+1):
        for gx in range(BORDER_BUFFER_CELLS):
            border_obs.add((gx, gy))
        for gx in range(max_g - BORDER_BUFFER_CELLS + 1, max_g+1):
            border_obs.add((gx, gy))
    # carve out goal zones
    for goals in GOAL_RANGE.values():
        if not goals: continue
        for goal_cm in goals:
            gx0, gy0 = cm_to_grid_coords(goal_cm[0], goal_cm[1])
            for dx in range(-BORDER_BUFFER_CELLS, BORDER_BUFFER_CELLS+1):
                for dy in range(-BORDER_BUFFER_CELLS, BORDER_BUFFER_CELLS+1):
                    border_obs.discard((gx0+dx, gy0+dy))
    return border_obs

def compute_best_route(balls_list, goal_name):
    if not balls_list:
        return [], []
    start_cm   = robot_position_cm or START_POINT_CM
    start_cell = cm_to_grid_coords(*start_cm)
    grid_w     = REAL_WIDTH_CM // GRID_SPACING_CM
    grid_h     = REAL_HEIGHT_CM // GRID_SPACING_CM

    ball_cells = [cm_to_grid_coords(b[0], b[1]) for b in balls_list]
    goal_cm    = GOAL_RANGE[goal_name][0]
    goal_cell  = cm_to_grid_coords(*goal_cm)

    points     = [start_cell] + ball_cells
    border_obs = get_border_buffer_obstacles()
    all_obs    = static_obstacles | dynamic_obstacles | border_obs
    exp_obs    = get_expanded_obstacles(all_obs)

    # distance+path matrix
    dm = {}
    for i,p in enumerate(points):
        for j,q in enumerate(points[i+1:], start=i+1):
            path = astar(p, q, grid_w, grid_h, exp_obs)
            dm[(i,j)] = (len(path), path)
            dm[(j,i)] = (len(path), path[::-1])

    # ball→goal
    bg = {}
    for idx, cell in enumerate(ball_cells, start=1):
        path = astar(cell, goal_cell, grid_w, grid_h, exp_obs)
        bg[idx] = (len(path), path)

    # brute‐force best pickup order
    best_cost = float('inf')
    best_perm = None
    for perm in itertools.permutations(range(1, len(ball_cells)+1)):
        cost = 0
        seq  = [0] + list(perm) + ['G']
        for a,b in zip(seq, seq[1:]):
            cost += bg[a][0] if b=='G' else dm[(a,b)][0]
        if cost < best_cost:
            best_cost, best_perm = cost, perm

    # assemble cm route & full grid path
    route_cm   = [start_cm] + [(balls_list[i-1][0], balls_list[i-1][1]) for i in best_perm] + [goal_cm]
    full_cells = []
    for a, b in zip([0]+list(best_perm), list(best_perm)+['G']):
        _, segment = (bg[a] if b=='G' else dm[(a,b)])
        full_cells += segment

    # smooth the path to reduce turns
    full_cells = prune_colinear(full_cells)
    return route_cm, full_cells

def create_and_cache_grid_overlay():
    global grid_overlay
    canvas = np.zeros((REAL_HEIGHT_CM+1, REAL_WIDTH_CM+1,3), dtype=np.uint8)
    for x in range(0, REAL_WIDTH_CM+1, GRID_SPACING_CM):
        cv2.line(canvas, (x,0), (x,REAL_HEIGHT_CM), GRID_LINE_COLOR, 1)
    for y in range(0, REAL_HEIGHT_CM+1, GRID_SPACING_CM):
        cv2.line(canvas, (0,y), (REAL_WIDTH_CM,y), GRID_LINE_COLOR, 1)
    w_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    grid_overlay = cv2.warpPerspective(canvas, homography_matrix, (w_px, h_px), flags=cv2.INTER_LINEAR)

def draw_metric_grid(frame):
    if grid_overlay is None:
        return frame
    blended = cv2.addWeighted(frame, 1.0, grid_overlay, 0.5, 0)
    for gx,gy in static_obstacles:
        x_cm, y_cm = gx*GRID_SPACING_CM, gy*GRID_SPACING_CM
        pt = np.array([[[x_cm,y_cm]]], dtype="float32")
        px, py = cv2.perspectiveTransform(pt, homography_matrix)[0][0]
        cv2.circle(blended, (int(px),int(py)), OBSTACLE_DRAW_RADIUS_PX, (0,0,255), -1)
    for gx,gy in dynamic_obstacles:
        x_cm, y_cm = gx*GRID_SPACING_CM, gy*GRID_SPACING_CM
        pt = np.array([[[x_cm,y_cm]]], dtype="float32")
        px, py = cv2.perspectiveTransform(pt, homography_matrix)[0][0]
        cv2.circle(blended, (int(px),int(py)), OBSTACLE_DRAW_RADIUS_PX, (0,165,255), -1)
    return blended

def draw_full_route(frame, ball_positions):
    global cached_route, last_ball_positions_cm, last_selected_goal, pending_route, full_grid_path
    if homography_matrix is None:
        return frame

    if robot_position_cm:
        pt = np.array([[[robot_position_cm[0],robot_position_cm[1]]]], dtype="float32")
        px, py = cv2.perspectiveTransform(pt, homography_matrix)[0][0]
        cv2.circle(frame, (int(px),int(py)), 12, (255,0,0), 3)
        cv2.putText(frame, "ROBOT", (int(px)-20,int(py)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    chosen = sorted(
        ball_positions,
        key=lambda b: abs(cm_to_grid_coords(*(robot_position_cm or START_POINT_CM))[0] - cm_to_grid_coords(b[0],b[1])[0])
                  + abs(cm_to_grid_coords(*(robot_position_cm or START_POINT_CM))[1] - cm_to_grid_coords(b[0],b[1])[1])
    )[:MAX_BALLS_TO_COLLECT]

    for _,_,lbl,cx,cy in chosen:
        cv2.circle(frame, (int(cx),int(cy)), 10, (0,255,0), 3)
        cv2.putText(frame, lbl, (int(cx)-10,int(cy)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    route_changed = (
        cached_route is None or
        len(chosen) != len(last_ball_positions_cm) or
        any(abs(x1-x2)>1 or abs(y1-y2)>1
            for (x1,y1, *_), (x2,y2, *_)
            in zip(chosen, last_ball_positions_cm)) or
        last_selected_goal != selected_goal
    )
    if route_changed:
        route_cm, grid_cells = compute_best_route(chosen, selected_goal)
        cached_route           = route_cm
        last_ball_positions_cm = chosen.copy()
        last_selected_goal     = selected_goal
        full_grid_path         = grid_cells
    else:
        route_cm = cached_route

    pending_route = route_cm
    overlay = frame.copy()
    total_cm = 0
    exp_obs = get_expanded_obstacles(static_obstacles | dynamic_obstacles | get_border_buffer_obstacles())

    for i in range(len(route_cm)-1):
        sc = cm_to_grid_coords(*route_cm[i])
        ec = cm_to_grid_coords(*route_cm[i+1])
        path = astar(sc, ec, REAL_WIDTH_CM//GRID_SPACING_CM, REAL_HEIGHT_CM//GRID_SPACING_CM, exp_obs)
        for j in range(1, len(path)):
            x0, y0 = path[j-1][0]*GRID_SPACING_CM, path[j-1][1]*GRID_SPACING_CM
            x1, y1 = path[j][0]*GRID_SPACING_CM,   path[j][1]*GRID_SPACING_CM
            p0 = cv2.perspectiveTransform(np.array([[[x0,y0]]], dtype="float32"), homography_matrix)[0][0]
            p1 = cv2.perspectiveTransform(np.array([[[x1,y1]]], dtype="float32"), homography_matrix)[0][0]
            cv2.line(overlay, (int(p0[0]),int(p0[1])), (int(p1[0]),int(p1[1])), PATH_COLOR, 3)
            total_cm += GRID_SPACING_CM

    cv2.putText(overlay, f"Total Path: {total_cm}cm to Goal {selected_goal}",
                (10, overlay.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    return overlay

def click_to_set_corners(event, x, y, flags, param):
    global calibration_points, homography_matrix, inv_homography_matrix
    if event == cv2.EVENT_LBUTTONDOWN and len(calibration_points) < 4:
        calibration_points.append([x, y])
        print(f"Corner {len(calibration_points)}: ({x},{y})")
        if len(calibration_points) == 4:
            dst = np.array([[0,0],[REAL_WIDTH_CM,0],
                            [REAL_WIDTH_CM,REAL_HEIGHT_CM],[0,REAL_HEIGHT_CM]], dtype="float32")
            src = np.array(calibration_points, dtype="float32")
            homography_matrix    = cv2.getPerspectiveTransform(dst, src)
            inv_homography_matrix = np.linalg.inv(homography_matrix)
            create_and_cache_grid_overlay()
            print("✅ Homography done.")
    elif event == cv2.EVENT_RBUTTONDOWN and homography_matrix is not None:
        pt_cm = pixel_to_cm(x, y)
        if pt_cm:
            gx, gy = cm_to_grid_coords(pt_cm[0], pt_cm[1])
            max_gx = REAL_WIDTH_CM // GRID_SPACING_CM
            max_gy = REAL_HEIGHT_CM // GRID_SPACING_CM
            if 0 <= gx <= max_gx and 0 <= gy <= max_gy:
                if (gx, gy) in static_obstacles:
                    static_obstacles.remove((gx, gy))
                    print(f"Removed obstacle at ({gx},{gy})")
                else:
                    static_obstacles.add((gx, gy))
                    print(f"Added obstacle at ({gx},{gy})")

def ensure_outer_edges_walkable():
    max_x = REAL_WIDTH_CM // GRID_SPACING_CM
    max_y = REAL_HEIGHT_CM // GRID_SPACING_CM
    for gx in range(max_x+1):
        static_obstacles.discard((gx, 0))
        static_obstacles.discard((gx, max_y))
    for gy in range(max_y+1):
        static_obstacles.discard((0, gy))
        static_obstacles.discard((max_x, gy))

# === Robot Communication ===
robot_sock = None
def init_robot_connection(ip, port, timeout=2.0):
    global robot_sock
    try:
        robot_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        robot_sock.settimeout(timeout)
        robot_sock.connect((ip, port))
        robot_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        print(f"Connected to robot at {ip}:{port}")
    except Exception as e:
        print("Robot connection failed:", e)
        robot_sock = None

def send_path(grid_path, heading):
    if robot_sock is None:
        print("No robot connection.")
        return
    payload = {"heading": heading, "path": [[gx,gy] for gx,gy in grid_path]}
    try:
        robot_sock.sendall((json.dumps(payload)+"\n").encode())
        print(f"Sent {len(grid_path)} cells")
    except Exception as e:
        print("Send failed:", e)

# === Thread Functions ===
def capture_frames():
    cnt = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        cnt += 1
        if cnt % SKIP_FRAMES == 0:
            try:
                frame_queue.put(frame, timeout=0.02)
            except:
                pass
    print("Capture exiting")

def process_frames():
    global dynamic_obstacles, robot_position_cm
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.02)
        except Empty:
            continue
        dynamic_obstacles.clear()
        orig = frame.copy()
        small = cv2.resize(frame, (416,416))
        h_full, w_full = orig.shape[:2]
        sx, sy = w_full/416, h_full/416

        # 1) Balls
        preds = model_v1.predict(small,
                confidence=int(CONFIDENCE_THRESHOLD*100),
                overlap=int(OVERLAP_THRESHOLD*100)).json()
        ball_positions_cm.clear()
        for d in preds.get('predictions', []):
            lbl = d['class'].strip().lower()
            if "ball" not in lbl:
                continue
            cx, cy = int(d['x']*sx), int(d['y']*sy)
            w, h  = int(d['width']*sx), int(d['height']*sy)
            color = class_colors.setdefault(lbl,
                        (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            x1, y1, x2, y2 = cx-w//2, cy-h//2, cx+w//2, cy+h//2
            cv2.rectangle(orig, (x1,y1), (x2,y2), color, 2)
            cv2.putText(orig, lbl, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cm = pixel_to_cm(cx, cy)
            if cm and not (IGNORED_AREA['x_min'] <= cm[0] <= IGNORED_AREA['x_max']
                           and IGNORED_AREA['y_min'] <= cm[1] <= IGNORED_AREA['y_max']):
                ball_positions_cm.append((cm[0], cm[1], lbl, cx, cy))

        # 2) Robot & obstacles
        preds = model_v3.predict(small,
                confidence=int(CONFIDENCE_THRESHOLD*100),
                overlap=int(OVERLAP_THRESHOLD*100)).json()
        for d in preds.get('predictions', []):
            lbl = d['class'].strip().lower()
            cx, cy = int(d['x']*sx), int(d['y']*sy)
            if "robot" in lbl:
                cm = pixel_to_cm(cx, cy)
                if cm:
                    robot_position_cm = cm
                continue
            if "ball" in lbl:
                continue
            w, h = int(d['width']*sx), int(d['height']*sy)
            color = class_colors.setdefault(lbl,
                        (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            x1, y1, x2, y2 = cx-w//2, cy-h//2, cx+w//2, cy+h//2
            cv2.rectangle(orig, (x1,y1), (x2,y2), color, 2)
            cv2.putText(orig, lbl, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if homography_matrix is not None:
                for px in range(x1, x2, 10):
                    for py in range(y1, y2, 10):
                        real = pixel_to_cm(px, py)
                        if real:
                            gx, gy = cm_to_grid_coords(real[0], real[1])
                            if 0 <= gx <= REAL_WIDTH_CM//GRID_SPACING_CM and 0 <= gy <= REAL_HEIGHT_CM//GRID_SPACING_CM:
                                dynamic_obstacles.add((gx, gy))

        # 3) Red‐cross dynamic obstacles
        if homography_matrix is not None:
            hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255]))
            m2 = cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
            red = cv2.bitwise_or(m1, m2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
            red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
            cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best, area = None, 0
            px_per_x, px_per_y = w_full/REAL_WIDTH_CM, h_full/REAL_HEIGHT_CM
            for c in cnts:
                a = cv2.contourArea(c)
                if a < MIN_RED_AREA_PX:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                ac = (bw/px_per_x)*(bh/px_per_y)
                if ac > MAX_RED_AREA_CM2:
                    continue
                if a > area:
                    best, area = c, a
            if best is not None:
                bx, by, bw, bh = cv2.boundingRect(best)
                for px in range(bx, bx+bw, 10):
                    for py in range(by, by+bh, 10):
                        if cv2.pointPolygonTest(best, (px,py), False) >= 0:
                            real = pixel_to_cm(px, py)
                            if real:
                                gx, gy = cm_to_grid_coords(real[0], real[1])
                                if 0 <= gx <= REAL_WIDTH_CM//GRID_SPACING_CM and 0 <= gy <= REAL_HEIGHT_CM//GRID_SPACING_CM:
                                    dynamic_obstacles.add((gx, gy))

        # 4) Draw everything
        frame_g = draw_metric_grid(orig)
        frame_r = draw_full_route(frame_g, ball_positions_cm)
        try:
            output_queue.put(frame_r, timeout=0.02)
        except:
            pass

    print("Processing exiting")

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
            if key == ord('1'):
                selected_goal = 'A'; print("Goal A")
            if key == ord('2'):
                selected_goal = 'B'; print("Goal B")
            if key == ord('s') and pending_route:
                save_route_to_file(pending_route)
                if full_grid_path:
                    send_path(full_grid_path, ROBOT_HEADING)
            continue

        cv2.imshow("Live Object Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break
        if key == ord('1'):
            selected_goal = 'A'; print("Goal A")
        if key == ord('2'):
            selected_goal = 'B'; print("Goal B")
        if key == ord('s') and pending_route:
            save_route_to_file(pending_route)
            if full_grid_path:
                send_path(full_grid_path, ROBOT_HEADING)
    print("Display exiting")

# === Main ===
if __name__ == "__main__":
    ensure_outer_edges_walkable()
    selected_goal = 'A'
    init_robot_connection(ROBOT_IP, ROBOT_PORT)

    t1 = threading.Thread(target=capture_frames)
    t2 = threading.Thread(target=process_frames)
    t3 = threading.Thread(target=display_frames)
    t1.start(); t2.start(); t3.start()
    t3.join(); t1.join(); t2.join()

    if robot_sock:
        robot_sock.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly")
