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

START_POINT_CM = (20, 20)         # fallback if robot not seen
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

ROBOT_IP      = "10.137.48.57"
ROBOT_PORT    = 12345
ROBOT_HEADING = "E"

OBSTACLE_DRAW_RADIUS_PX = 6
GRID_LINE_COLOR         = (100, 100, 100)
PATH_COLOR              = (0, 255, 255)
TEXT_COLOR              = (0, 255, 255)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# === END CONFIGURABLE CONSTANTS ===

# === Roboflow Model Initialization ===
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
model_v1 = project.version(VERSION).model  # for ball detection
model_v3 = project.version(VERSION).model  # for obstacles + robot

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
obstacles               = set()
ball_positions_cm       = []
last_ball_positions_cm  = []
cached_route            = None
last_selected_goal      = None
pending_route           = None
full_grid_path          = []

# === Robot Start Point (dynamic) ===
robot_position_cm = None

# === Color Mapping for Classes ===
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
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx <= grid_w and 0 <= ny <= grid_h and (nx, ny) not in obstacles_set:
                tentative = g_score[current] + 1
                if tentative < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative
                    f = tentative + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, (nx, ny)))
    return []

def significant_change(balls, last, tol_cm=1.0):
    if len(balls) != len(last):
        return True
    for (x,y,l,_,_), (lx,ly,ll,_,_) in zip(balls, last):
        if l != ll or abs(x-lx)>tol_cm or abs(y-ly)>tol_cm:
            return True
    return False

def pick_top_n(balls, n=MAX_BALLS_TO_COLLECT):
    # use robot_position_cm as start, else fallback
    start_cm = robot_position_cm or START_POINT_CM
    sx, sy = cm_to_grid_coords(*start_cm)
    def dist(b):
        bx, by = b[0], b[1]
        gx, gy = cm_to_grid_coords(bx, by)
        return abs(sx-gx)+abs(sy-gy)
    return sorted(balls, key=dist)[:n]

def get_expanded_obstacles(raw):
    exp = set()
    max_g = REAL_WIDTH_CM // GRID_SPACING_CM
    max_h = REAL_HEIGHT_CM // GRID_SPACING_CM
    for (gx,gy) in raw:
        for dx in range(-BUFFER_CELLS, BUFFER_CELLS+1):
            for dy in range(-BUFFER_CELLS, BUFFER_CELLS+1):
                nx, ny = gx+dx, gy+dy
                if 0<=nx<=max_g and 0<=ny<=max_h:
                    exp.add((nx,ny))
    return exp

def compute_best_route(balls_list, goal_name):
    if not balls_list:
        return [], []
    # determine start cell from robot_position_cm
    start_cm = robot_position_cm or START_POINT_CM
    start_cell = cm_to_grid_coords(*start_cm)

    grid_w = REAL_WIDTH_CM // GRID_SPACING_CM
    grid_h = REAL_HEIGHT_CM // GRID_SPACING_CM
    ball_cells = [cm_to_grid_coords(b[0], b[1]) for b in balls_list]
    goal_cm = GOAL_RANGE[goal_name][0]
    goal_cell = cm_to_grid_coords(goal_cm[0], goal_cm[1])

    points = [start_cell] + ball_cells
    exp_obs = get_expanded_obstacles(obstacles)

    # precompute pairwise
    dm = {}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            p, q = points[i], points[j]
            path = astar(p, q, grid_w, grid_h, exp_obs)
            dm[(i,j)] = (len(path), path)
            dm[(j,i)] = (len(path), path[::-1])

    # ball->goal
    bg = {}
    for idx, cell in enumerate(ball_cells, start=1):
        path = astar(cell, goal_cell, grid_w, grid_h, exp_obs)
        bg[idx] = (len(path), path)

    best = None
    best_cost = float('inf')
    for perm in itertools.permutations(range(1, len(ball_cells)+1)):
        cost = 0
        segs = []
        # start->first
        c,p = dm[(0, perm[0])]
        cost+=c; segs.append(p)
        # between balls
        for a,b in zip(perm, perm[1:]):
            c,p = dm[(a,b)]
            cost+=c; segs.append(p)
        # last->goal
        c,p = bg[perm[-1]]
        cost+=c; segs.append(p)

        if cost < best_cost:
            best_cost, best = cost, perm

    # reconstruct few
    route_cm = [start_cm] + [(balls_list[i-1][0], balls_list[i-1][1]) for i in best] + [goal_cm]
    full_cells = []
    # rebuild path cells
    seq = [0] + list(best) + ['G']
    for i in range(len(seq)-1):
        a, b = seq[i], seq[i+1]
        if b == 'G':
            _, path = bg[a]
        else:
            path = dm[(a,b)][1]
        full_cells.extend(path)

    return route_cm, full_cells

def create_and_cache_grid_overlay():
    global grid_overlay
    canvas = np.zeros((REAL_HEIGHT_CM+1, REAL_WIDTH_CM+1,3), dtype=np.uint8)
    for x in range(0, REAL_WIDTH_CM+1, GRID_SPACING_CM):
        cv2.line(canvas,(x,0),(x,REAL_HEIGHT_CM),GRID_LINE_COLOR,1)
    for y in range(0, REAL_HEIGHT_CM+1, GRID_SPACING_CM):
        cv2.line(canvas,(0,y),(REAL_WIDTH_CM,y),GRID_LINE_COLOR,1)
    w_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    grid_overlay = cv2.warpPerspective(canvas, homography_matrix, (w_px,h_px), flags=cv2.INTER_LINEAR)
    print("🗺️ Cached grid overlay.")

def draw_metric_grid(frame):
    if grid_overlay is None:
        return frame
    blended = cv2.addWeighted(frame,1.0,grid_overlay,0.5,0)
    for (gx,gy) in obstacles:
        x_cm, y_cm = gx*GRID_SPACING_CM, gy*GRID_SPACING_CM
        pt = np.array([[[x_cm,y_cm]]], dtype="float32")
        px, py = cv2.perspectiveTransform(pt, homography_matrix)[0][0]
        cv2.circle(blended,(int(px),int(py)), OBSTACLE_DRAW_RADIUS_PX,(0,0,255),-1)
    return blended

def draw_full_route(frame, ball_positions):
    global cached_route, last_ball_positions_cm, last_selected_goal, pending_route, full_grid_path
    if homography_matrix is None:
        return frame

    # mark robot on screen
    if robot_position_cm:
        pt = np.array([[[robot_position_cm[0],robot_position_cm[1]]]], dtype="float32")
        px, py = cv2.perspectiveTransform(pt, homography_matrix)[0][0]
        cv2.circle(frame,(int(px),int(py)),12,(255,0,0),3)
        cv2.putText(frame,"ROBOT",(int(px)-20,int(py)-20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)

    chosen = pick_top_n(ball_positions)
    for (_,_,lbl,cx,cy) in chosen:
        cv2.circle(frame,(int(cx),int(cy)),10,(0,255,0),3)
        cv2.putText(frame,lbl,(int(cx)-10,int(cy)-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    route_changed = (
        cached_route is None
        or significant_change(chosen, last_ball_positions_cm)
        or last_selected_goal != selected_goal
    )

    if route_changed:
        if len(ball_positions) > MAX_BALLS_TO_COLLECT:
            coords = [(round(b[0],1),round(b[1],1),b[2]) for b in chosen]
            print(f"📋 Using only these {MAX_BALLS_TO_COLLECT} balls: {coords}")
        route_cm, grid_cells = compute_best_route(chosen, selected_goal)
        cached_route            = route_cm
        last_ball_positions_cm  = chosen.copy()
        last_selected_goal      = selected_goal
        full_grid_path          = grid_cells
    else:
        route_cm = cached_route

    pending_route = route_cm

    overlay = frame.copy()
    total_cm = 0
    grid_w = REAL_WIDTH_CM // GRID_SPACING_CM
    grid_h = REAL_HEIGHT_CM // GRID_SPACING_CM

    for i in range(len(route_cm)-1):
        sc = cm_to_grid_coords(*route_cm[i])
        ec = cm_to_grid_coords(*route_cm[i+1])
        path = astar(sc, ec, grid_w, grid_h, get_expanded_obstacles(obstacles))
        if not path:
            continue
        for j in range(len(path)):
            prev = path[j-1] if j>0 else path[0]
            curr = path[j]
            x0, y0 = prev[0]*GRID_SPACING_CM, prev[1]*GRID_SPACING_CM
            x1, y1 = curr[0]*GRID_SPACING_CM, curr[1]*GRID_SPACING_CM
            p0 = cv2.perspectiveTransform(np.array([[[x0,y0]]],dtype="float32"), homography_matrix)[0][0]
            p1 = cv2.perspectiveTransform(np.array([[[x1,y1]]],dtype="float32"), homography_matrix)[0][0]
            cv2.line(overlay,(int(p0[0]),int(p0[1])),(int(p1[0]),int(p1[1])),PATH_COLOR,3)
            total_cm += GRID_SPACING_CM

    cv2.putText(overlay,f"Total Path: {total_cm}cm to Goal {selected_goal}",
                (10, overlay.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,TEXT_COLOR,2)
    return overlay

def click_to_set_corners(event, x, y, flags, param):
    global calibration_points, homography_matrix, inv_homography_matrix, grid_overlay, obstacles
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append([x, y])
            print(f"Corner {len(calibration_points)} set: ({x}, {y})")
        if len(calibration_points) == 4 and homography_matrix is None:
            dst = np.array([[0,0],[REAL_WIDTH_CM,0],
                            [REAL_WIDTH_CM,REAL_HEIGHT_CM],[0,REAL_HEIGHT_CM]],dtype="float32")
            src = np.array(calibration_points,dtype="float32")
            homography_matrix    = cv2.getPerspectiveTransform(dst, src)
            inv_homography_matrix = np.linalg.inv(homography_matrix)
            print("✅ Homography calculated.")
            create_and_cache_grid_overlay()

    elif event == cv2.EVENT_RBUTTONDOWN and homography_matrix is not None:
        pt_cm = pixel_to_cm(x, y)
        if pt_cm:
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
    for gx in range(max_x+1):
        obstacles.discard((gx, 0))
        obstacles.discard((gx, max_y))
    for gy in range(max_y+1):
        obstacles.discard((0, gy))
        obstacles.discard((max_x, gy))
    print("✅ Outer edges cleared.")

robot_sock = None

def init_robot_connection(ip: str, port: int, timeout=2.0):
    global robot_sock
    try:
        robot_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        robot_sock.settimeout(timeout)
        robot_sock.connect((ip, port))
        robot_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        print(f"📡 Connected to robot at {ip}:{port}")
    except Exception as e:
        print(f"❌ Could not connect to robot: {e}")
        robot_sock = None

def send_path(grid_path: list, heading: str):
    global robot_sock
    if robot_sock is None:
        print("⚠️  No robot connection, aborting send.")
        return
    try:
        payload = {
            "heading": heading,
            "path": [[int(gx), int(gy)] for (gx, gy) in grid_path]
        }
        data = json.dumps(payload).encode("utf-8")
        prefix = len(data).to_bytes(4, "big")
        robot_sock.sendall(prefix + data)
        print(f"📨 Sent path → {len(grid_path)} cells, heading={heading}")
    except Exception as e:
        print(f"❌ Failed to send path: {e}")

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
    print("📷 capture_frames exiting")

def process_frames():
    global ball_positions_cm, obstacles, robot_position_cm
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

        # 1) DETECT BALLS
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
            cx = int(d['x'] * scale_x)
            cy = int(d['y'] * scale_y)
            w  = int(d['width']  * scale_x)
            h  = int(d['height'] * scale_y)
            color = (200,200,255) if "white" in lbl else (
                    (0,165,255) if "orange" in lbl else
                    class_colors.setdefault(lbl, (random.randint(0,255),
                                                  random.randint(0,255),
                                                  random.randint(0,255))))
            x1,y1 = cx-w//2, cy-h//2
            x2,y2 = cx+w//2, cy+h//2
            cv2.rectangle(original,(x1,y1),(x2,y2),color,2)
            cv2.putText(original,lbl,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            cm = pixel_to_cm(cx, cy)
            if cm and not (
                IGNORED_AREA['x_min'] <= cm[0] <= IGNORED_AREA['x_max'] and
                IGNORED_AREA['y_min'] <= cm[1] <= IGNORED_AREA['y_max']
            ):
                ball_positions_cm.append((cm[0],cm[1],lbl,cx,cy))

        # 2) DETECT ROBOT & OTHER OBSTACLES
        preds_v3 = model_v3.predict(
            small,
            confidence=int(CONFIDENCE_THRESHOLD * 100),
            overlap=int(OVERLAP_THRESHOLD * 100)
        ).json()
        for d in preds_v3.get('predictions', []):
            lbl = d['class'].strip().lower()
            cx = int(d['x'] * scale_x)
            cy = int(d['y'] * scale_y)

            # record robot position only
            if "robot" in lbl:
                cm = pixel_to_cm(cx, cy)
                if cm:
                    robot_position_cm = cm
                continue

            # skip balls here so they never become obstacles
            if "ball" in lbl:
                continue

            # draw non-robot, non-ball boxes
            w  = int(d['width']  * scale_x)
            h  = int(d['height'] * scale_y)
            color = class_colors.setdefault(lbl, (random.randint(0,255),
                                                  random.randint(0,255),
                                                  random.randint(0,255)))
            x1,y1 = cx-w//2, cy-h//2
            x2,y2 = cx+w//2, cy+h//2
            cv2.rectangle(original,(x1,y1),(x2,y2),color,2)
            cv2.putText(original,lbl,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

            # sample inside for obstacles
            if homography_matrix is not None:
                max_gx = REAL_WIDTH_CM // GRID_SPACING_CM
                max_gy = REAL_HEIGHT_CM // GRID_SPACING_CM
                for sx in range(x1, x2, 10):
                    for sy in range(y1, y2, 10):
                        real = pixel_to_cm(sx, sy)
                        if not real:
                            continue
                        gx, gy = cm_to_grid_coords(real[0], real[1])
                        if 0 <= gx <= max_gx and 0 <= gy <= max_gy:
                            obstacles.add((gx, gy))

        # 3) RED CROSS (unchanged)
        if homography_matrix is not None:
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255]))
            mask2 = cv2.inRange(hsv, np.array([170,120,70]),np.array([180,255,255]))
            red_mask = cv2.bitwise_or(mask1, mask2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
            red_mask = cv2.morphologyEx(red_mask,cv2.MORPH_CLOSE,kernel)
            red_mask = cv2.morphologyEx(red_mask,cv2.MORPH_OPEN,kernel)
            contours,_ = cv2.findContours(red_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            best_cnt, best_area = None, 0
            px_per_x = w_full / REAL_WIDTH_CM
            px_per_y = h_full / REAL_HEIGHT_CM

            for cnt in contours:
                area_px = cv2.contourArea(cnt)
                if area_px < MIN_RED_AREA_PX:
                    continue
                x_r,y_r,w_r,h_r = cv2.boundingRect(cnt)
                area_cm = (w_r/px_per_x)*(h_r/px_per_y)
                if area_cm > MAX_RED_AREA_CM2:
                    continue
                if area_px > best_area:
                    best_cnt, best_area = cnt, area_px

            new_obs = set()
            if best_cnt is not None:
                bx,by,bw_cnt,bh_cnt = cv2.boundingRect(best_cnt)
                for sx in range(bx, bx+bw_cnt, 10):
                    for sy in range(by, by+bh_cnt, 10):
                        if cv2.pointPolygonTest(best_cnt,(sx,sy),False) >= 0:
                            real = pixel_to_cm(sx, sy)
                            if not real: continue
                            gx, gy = cm_to_grid_coords(real[0], real[1])
                            max_gx = REAL_WIDTH_CM//GRID_SPACING_CM
                            max_gy = REAL_HEIGHT_CM//GRID_SPACING_CM
                            if 0<=gx<=max_gx and 0<=gy<=max_gy:
                                new_obs.add((gx,gy))
            obstacles |= new_obs

        # 4) DRAW GRID & ROUTE
        frame_grid  = draw_metric_grid(original)
        frame_route = draw_full_route(frame_grid, ball_positions_cm)

        try:
            output_queue.put(frame_route, timeout=0.02)
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
                selected_goal = 'A'; print("✅ Selected Goal A")
            elif key == ord('2'):
                selected_goal = 'B'; print("✅ Selected Goal B")
            elif key == ord('s'):
                if pending_route:
                    save_route_to_file(pending_route)
                if full_grid_path:
                    send_path(full_grid_path, ROBOT_HEADING)
            continue

        cv2.imshow("Live Object Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break
        elif key == ord('1'):
            selected_goal = 'A'; print("✅ Selected Goal A")
        elif key == ord('2'):
            selected_goal = 'B'; print("✅ Selected Goal B")
        elif key == ord('s'):
            if pending_route:
                save_route_to_file(pending_route)
            if full_grid_path:
                send_path(ROBOT_IP, ROBOT_PORT, full_grid_path, ROBOT_HEADING)

    print("🖼️ display_frames exiting")

# === Main Thread ===
if __name__ == "__main__":
    ensure_outer_edges_walkable()
    selected_goal = 'A'

    # open persistent robot connection once
    init_robot_connection(ROBOT_IP, ROBOT_PORT)

    cap_thread = threading.Thread(target=capture_frames)
    proc_thread = threading.Thread(target=process_frames)
    disp_thread = threading.Thread(target=display_frames)

    cap_thread.start()
    proc_thread.start()
    disp_thread.start()

    disp_thread.join()
    cap_thread.join()
    proc_thread.join()

    # clean up robot socket
    if robot_sock:
        robot_sock.close()
    cap.release()
    cv2.destroyAllWindows()
    print("✂️ Exiting cleanly")