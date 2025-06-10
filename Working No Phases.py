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

ROBOFLOW_API_KEY = "7kMjalIwU9TqGmKM0g4i"
WORKSPACE_NAME   = "pingpong-fafrv"
PROJECT_NAME     = "newpingpongdetector"
VERSION          = 1

CAMERA_INDEX     = 1
FRAME_WIDTH      = 1280
FRAME_HEIGHT     = 720
FRAMES_PER_SEC   = 30
BUFFER_SIZE      = 1

REAL_WIDTH_CM    = 180
REAL_HEIGHT_CM   = 120
GRID_SPACING_CM  = 2

START_POINT_CM = (20, 20)
GOAL_A_CM      = (REAL_WIDTH_CM, REAL_HEIGHT_CM // 2)
GOAL_B_CM      = None

GOAL_RANGE = {
    'A': [GOAL_A_CM],
    'B': GOAL_B_CM
}

OBSTACLE_BUFFER_CM = 10
BUFFER_CELLS       = int(np.ceil(OBSTACLE_BUFFER_CM / GRID_SPACING_CM))

MAX_BALLS_TO_COLLECT = 4

IGNORED_AREA = {
    'x_min': 50, 'x_max': 100,
    'y_min': 50, 'y_max': 100
}

CONFIDENCE_THRESHOLD = 0.50
OVERLAP_THRESHOLD    = 0.05
MIN_RED_AREA_PX      = 500
MAX_RED_AREA_CM2     = 400

SKIP_FRAMES = 3

ROBOT_IP      = "10.225.58.57"
ROBOT_PORT    = 12345
ROBOT_HEADING = "N"

OBSTACLE_DRAW_RADIUS_PX = 6
GRID_LINE_COLOR         = (100, 100, 100)
PATH_COLOR              = (0, 255, 255)
TEXT_COLOR              = (0, 255, 255)

RANDOM_SEED = 42

# === MODEL INITIALIZATION ===

rf       = Roboflow(api_key=ROBOFLOW_API_KEY)
project  = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
model_v1 = project.version(VERSION).model
model_v3 = project.version(VERSION).model

# === VIDEO CAPTURE SETUP ===

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
cap.set(cv2.CAP_PROP_FPS, FRAMES_PER_SEC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# === THREADING & QUEUES ===

frame_queue  = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
stop_event   = threading.Event()

# === HOMOGRAPHY & STATE ===

calibration_points    = []
homography_matrix     = None
inv_homography_matrix = None
grid_overlay          = None

obstacles               = set()
ball_positions_cm       = []
last_ball_positions_cm  = []
cached_route            = None
last_selected_goal      = None
pending_route           = None
full_grid_path          = []

random.seed(RANDOM_SEED)
class_colors = {}

# === UTILITY FUNCTIONS ===

def save_route_to_file(route_cm, filename="route.txt"):
    with open(filename, "w") as f:
        for x, y in route_cm:
            f.write(f"{x:.2f},{y:.2f}\n")
    print("📦 Route saved to route.txt")

def pixel_to_cm(px, py):
    if inv_homography_matrix is None:
        return None
    pt      = np.array([[[px, py]]], dtype="float32")
    real_pt = cv2.perspectiveTransform(pt, inv_homography_matrix)[0][0]
    return real_pt[0], real_pt[1]

def cm_to_grid_coords(x_cm, y_cm):
    return int(x_cm // GRID_SPACING_CM), int(y_cm // GRID_SPACING_CM)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, grid_w, grid_h, obstacles_set):
    open_set  = [(0, start)]
    came_from = {}
    g_score   = {start: 0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        x, y = current
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx <= grid_w and 0 <= ny <= grid_h and (nx, ny) not in obstacles_set:
                tent = g_score[current] + 1
                if tent < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)]   = tent
                    came_from[(nx, ny)] = current
                    f = tent + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, (nx, ny)))
    return []

def significant_change(ball_positions, last_positions, tol_cm=1.0):
    if len(ball_positions) != len(last_positions):
        return True
    for (x, y, lbl, _, _), (lx, ly, llbl, _, _) in zip(ball_positions, last_positions):
        if lbl != llbl or abs(x-lx)>tol_cm or abs(y-ly)>tol_cm:
            return True
    return False

def pick_top_n(balls, start_cm, n=MAX_BALLS_TO_COLLECT):
    sx, sy = cm_to_grid_coords(*start_cm)
    def md(ball):
        bx, by = cm_to_grid_coords(ball[0], ball[1])
        return abs(sx-bx)+abs(sy-by)
    return sorted(balls, key=md)[:n]

def get_expanded_obstacles(raw_obstacles):
    exp = set()
    for gx, gy in raw_obstacles:
        for dx in range(-BUFFER_CELLS, BUFFER_CELLS+1):
            for dy in range(-BUFFER_CELLS, BUFFER_CELLS+1):
                nx, ny = gx+dx, gy+dy
                if 0 <= nx <= REAL_WIDTH_CM//GRID_SPACING_CM and 0 <= ny <= REAL_HEIGHT_CM//GRID_SPACING_CM:
                    exp.add((nx, ny))
    return exp

def compute_best_route(balls_list, goal_name):
    if not balls_list:
        return [], []
    gw = REAL_WIDTH_CM//GRID_SPACING_CM
    gh = REAL_HEIGHT_CM//GRID_SPACING_CM
    start = cm_to_grid_coords(*START_POINT_CM)
    ball_cells = [cm_to_grid_coords(b[0], b[1]) for b in balls_list]
    goal_cm = GOAL_RANGE[goal_name][0]
    goal    = cm_to_grid_coords(goal_cm[0], goal_cm[1])
    points  = [start] + ball_cells
    exp_obs = get_expanded_obstacles(obstacles)

    dist_map = {}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            pth = astar(points[i], points[j], gw, gh, exp_obs)
            c   = len(pth)
            dist_map[(i,j)] = (c, pth)
            dist_map[(j,i)] = (c, pth[::-1])

    b2g = {}
    for idx, bc in enumerate(ball_cells, start=1):
        p = astar(bc, goal, gw, gh, exp_obs)
        b2g[idx] = (len(p), p)

    best_cost = None
    best_perm = None
    best_segs = None

    for perm in itertools.permutations(range(1, len(ball_cells)+1)):
        total = 0; segs=[]
        c,p   = dist_map[(0, perm[0])]
        total+=c; segs.append(p)
        for k in range(len(perm)-1):
            c,p = dist_map[(perm[k], perm[k+1])]
            total+=c; segs.append(p)
        c,p = b2g[perm[-1]]
        total+=c; segs.append(p)
        if best_cost is None or total<best_cost:
            best_cost, best_perm, best_segs = total, perm, segs

    route_cm = [START_POINT_CM] + [(balls_list[i-1][0], balls_list[i-1][1]) for i in best_perm] + [goal_cm]
    full = [cell for seg in best_segs for cell in seg]
    return route_cm, full

def create_and_cache_grid_overlay():
    global grid_overlay
    canvas = np.zeros((REAL_HEIGHT_CM+1, REAL_WIDTH_CM+1,3),dtype=np.uint8)
    for x in range(0, REAL_WIDTH_CM+1, GRID_SPACING_CM):
        cv2.line(canvas,(x,0),(x,REAL_HEIGHT_CM),GRID_LINE_COLOR,1)
    for y in range(0, REAL_HEIGHT_CM+1, GRID_SPACING_CM):
        cv2.line(canvas,(0,y),(REAL_WIDTH_CM,y),GRID_LINE_COLOR,1)
    h_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    grid_overlay = cv2.warpPerspective(canvas, homography_matrix, (w_px, h_px), flags=cv2.INTER_LINEAR)

def draw_metric_grid(frame):
    if grid_overlay is None:
        return frame
    blended = cv2.addWeighted(frame,1.0,grid_overlay,0.5,0)
    for gx, gy in obstacles:
        pt = np.array([[[gx*GRID_SPACING_CM, gy*GRID_SPACING_CM]]],dtype="float32")
        px = cv2.perspectiveTransform(pt,homography_matrix)[0][0]
        cv2.circle(blended,(int(px[0]),int(px[1])),OBSTACLE_DRAW_RADIUS_PX,(0,0,255),-1)
    return blended

def draw_full_route(frame, ball_positions):
    global cached_route, last_ball_positions_cm, last_selected_goal, pending_route, full_grid_path
    if homography_matrix is None:
        return frame

    chosen = pick_top_n(ball_positions, START_POINT_CM)
    for cx_cm, cy_cm, lbl, cx_px, cy_px in chosen:
        cv2.circle(frame,(int(cx_px),int(cy_px)),10,(0,255,0),3)
        # typo fixed: use cy_px instead of cx_py
        cv2.putText(frame, lbl, (int(cx_px)-10, int(cy_px)-15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    changed = (cached_route is None or
               significant_change(chosen, last_ball_positions_cm) or
               last_selected_goal != selected_goal)
    if changed:
        route_cm, cells = compute_best_route(chosen, selected_goal)
        cached_route           = route_cm
        last_ball_positions_cm = chosen.copy()
        last_selected_goal     = selected_goal
        full_grid_path         = cells
    else:
        route_cm = cached_route

    pending_route = route_cm

    overlay = frame.copy()
    total   = 0
    gw, gh  = REAL_WIDTH_CM//GRID_SPACING_CM, REAL_HEIGHT_CM//GRID_SPACING_CM
    for i in range(len(route_cm)-1):
        sc = cm_to_grid_coords(*route_cm[i])
        ec = cm_to_grid_coords(*route_cm[i+1])
        path = astar(sc, ec, gw, gh, get_expanded_obstacles(obstacles))
        for k in range(1,len(path)):
            p0,p1 = path[k-1], path[k]
            pt0 = np.array([[[p0[0]*GRID_SPACING_CM,p0[1]*GRID_SPACING_CM]]],dtype="float32")
            pt1 = np.array([[[p1[0]*GRID_SPACING_CM,p1[1]*GRID_SPACING_CM]]],dtype="float32")
            px0 = cv2.perspectiveTransform(pt0,homography_matrix)[0][0]
            px1 = cv2.perspectiveTransform(pt1,homography_matrix)[0][0]
            cv2.line(overlay,
                     (int(px0[0]),int(px0[1])),
                     (int(px1[0]),int(px1[1])),
                     PATH_COLOR,3)
            total += GRID_SPACING_CM

    cv2.putText(overlay,
                f"Total Path: {total}cm to Goal {selected_goal}",
                (10, overlay.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,TEXT_COLOR,2)
    return overlay

def click_to_set_corners(event, x, y, flags, param):
    global calibration_points, homography_matrix, inv_homography_matrix, grid_overlay, obstacles
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append([x, y])
        if len(calibration_points) == 4 and homography_matrix is None:
            dst = np.array([[0,0],[REAL_WIDTH_CM,0],[REAL_WIDTH_CM,REAL_HEIGHT_CM],[0,REAL_HEIGHT_CM]],dtype="float32")
            src = np.array(calibration_points,dtype="float32")
            homography_matrix    = cv2.getPerspectiveTransform(dst, src)
            inv_homography_matrix = np.linalg.inv(homography_matrix)
            create_and_cache_grid_overlay()
    elif event == cv2.EVENT_RBUTTONDOWN and homography_matrix is not None:
        real = pixel_to_cm(x, y)
        if real:
            gx, gy = cm_to_grid_coords(real[0], real[1])
            max_gx = REAL_WIDTH_CM//GRID_SPACING_CM
            max_gy = REAL_HEIGHT_CM//GRID_SPACING_CM
            if 0<=gx<=max_gx and 0<=gy<=max_gy:
                if (gx,gy) in obstacles:
                    obstacles.remove((gx,gy))
                else:
                    obstacles.add((gx,gy))

def ensure_outer_edges_walkable():
    max_x = REAL_WIDTH_CM//GRID_SPACING_CM
    max_y = REAL_HEIGHT_CM//GRID_SPACING_CM
    for gx in range(max_x+1):
        obstacles.discard((gx,0)); obstacles.discard((gx,max_y))
    for gy in range(max_y+1):
        obstacles.discard((0,gy)); obstacles.discard((max_x,gy))

def send_path(ip, port, grid_path, heading):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            sock.connect((ip, port))
            payload = {"heading": heading, "path": [[int(gx),int(gy)] for gx,gy in grid_path]}
            data = json.dumps(payload).encode("utf-8")
            sock.sendall(len(data).to_bytes(4,'big') + data)
            print(f"📨 Sent path to {ip}:{port}")
    except Exception as e:
        print(f"❌ Failed to send path: {e}")

def capture_frames():
    count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % SKIP_FRAMES == 0:
            try:
                frame_queue.put(frame, timeout=0.02)
            except:
                pass

def process_frames():
    global ball_positions_cm, obstacles
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.02)
        except Empty:
            continue

        orig = frame.copy()
        small = cv2.resize(frame, (416,416))
        h_full, w_full = orig.shape[:2]
        sx, sy = w_full/416, h_full/416

        # 1) BALL DETECTION
        preds_v1 = model_v1.predict(small,
                        confidence=int(CONFIDENCE_THRESHOLD*100),
                        overlap=int(OVERLAP_THRESHOLD*100)).json()
        ball_positions_cm.clear()
        for d in preds_v1.get('predictions', []):
            lbl = d['class'].strip().lower()
            if "ball" not in lbl: continue
            cx, cy = int(d['x']*sx), int(d['y']*sy)
            w, h   = int(d['width']*sx), int(d['height']*sy)
            x1,y1,x2,y2 = cx-w//2, cy-h//2, cx+w//2, cy+h//2
            if "white" in lbl: color=(200,200,255)
            elif "orange" in lbl: color=(0,165,255)
            else: color=class_colors.setdefault(lbl,(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
            cv2.rectangle(orig,(x1,y1),(x2,y2),color,2)
            cv2.putText(orig,lbl,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            real = pixel_to_cm(cx,cy)
            if real and not (IGNORED_AREA['x_min']<=real[0]<=IGNORED_AREA['x_max'] and IGNORED_AREA['y_min']<=real[1]<=IGNORED_AREA['y_max']):
                ball_positions_cm.append((real[0],real[1],lbl,cx,cy))

        # 2) OTHER OBJECTS → OBSTACLES (skip robot)
        preds_v3 = model_v3.predict(small,
                        confidence=int(CONFIDENCE_THRESHOLD*100),
                        overlap=int(OVERLAP_THRESHOLD*100)).json()
        for d in preds_v3.get('predictions', []):
            lbl = d['class'].strip().lower()
            if "ball" in lbl: continue
            cx, cy = int(d['x']*sx), int(d['y']*sy)
            w, h   = int(d['width']*sx), int(d['height']*sy)
            x1,y1,x2,y2 = cx-w//2, cy-h//2, cx+w//2, cy+h//2
            color = class_colors.setdefault(lbl,(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
            cv2.rectangle(orig,(x1,y1),(x2,y2),color,2)
            cv2.putText(orig,lbl,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            if "robot" in lbl:
                continue
            if homography_matrix is not None:
                max_gx = REAL_WIDTH_CM//GRID_SPACING_CM
                max_gy = REAL_HEIGHT_CM//GRID_SPACING_CM
                for sx_px in range(x1,x2,10):
                    for sy_px in range(y1,y2,10):
                        real = pixel_to_cm(sx_px,sy_px)
                        if not real: continue
                        gx, gy = cm_to_grid_coords(real[0],real[1])
                        if 0<=gx<=max_gx and 0<=gy<=max_gy:
                            obstacles.add((gx,gy))

        # 3) RED-CROSS DETECTION (unchanged)...

        # 4) DRAW & ROUTE
        grid_fr  = draw_metric_grid(orig)
        route_fr = draw_full_route(grid_fr, ball_positions_cm)
        try:
            output_queue.put(route_fr, timeout=0.02)
        except:
            pass

def display_frames():
    global selected_goal
    cv2.namedWindow("Live Object Detection")
    cv2.setMouseCallback("Live Object Detection", click_to_set_corners)
    while not stop_event.is_set():
        try:
            frame = output_queue.get(timeout=0.02)
        except Empty:
            if cv2.waitKey(1)&0xFF == ord('q'):
                stop_event.set()
            continue
        cv2.imshow("Live Object Detection", frame)
        key = cv2.waitKey(1)&0xFF
        if key == ord('q'):
            stop_event.set()
        elif key == ord('1'):
            selected_goal = 'A'
        elif key == ord('2'):
            selected_goal = 'B'
        elif key == ord('s'):
            if pending_route:
                save_route_to_file(pending_route)
            if full_grid_path:
                send_path(ROBOT_IP, ROBOT_PORT, full_grid_path, ROBOT_HEADING)

if __name__ == "__main__":
    ensure_outer_edges_walkable()
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
