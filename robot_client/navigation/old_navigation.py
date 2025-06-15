import cv2
import numpy as np
import itertools
import heapq

from config import (
    REAL_WIDTH_CM, REAL_HEIGHT_CM,
    GRID_SPACING_CM, START_POINT_CM,
    GOAL_RANGE, BUFFER_CELLS,
    OBSTACLE_DRAW_RADIUS_PX, GRID_LINE_COLOR,
    PATH_COLOR, TEXT_COLOR, MAX_BALLS_TO_COLLECT
)
from utils import significant_change


# === Global Shared State ===
homography_matrix = None
inv_homography_matrix = None
grid_overlay = None
calibration_points = []
obstacles = set()
cached_route = None
last_selected_goal = None
last_ball_positions_cm = []
full_grid_path = []
robot_position_cm = None
pending_route = None
selected_goal = 'A'

def get_aruco_robot_position_and_heading(frame):
    """
    Detects the ArUco marker ID 100 in the frame, draws it,
    and returns (x_cm, y_cm, heading_deg) in real‐world coords,
    or None if not found.
    """
    # make sure cv2.aruco is available
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id != 100:
                continue
            pts = corners[idx][0]  # 4×2
            # compute heading
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            angle_rad = np.arctan2(-dy, dx)
            heading_deg = (np.degrees(angle_rad) + 360) % 360
            # compute centroid
            cx = pts[:,0].mean()
            cy = pts[:,1].mean()
            # map to real‐world
            if inv_homography_matrix is not None:
                pt = np.array([[[cx, cy]]], dtype="float32")
                real_pt = cv2.perspectiveTransform(pt, inv_homography_matrix)[0][0]
                return real_pt[0], real_pt[1], heading_deg
    return None

# === Coordinate Conversion ===

def pixel_to_cm(px, py):
    if inv_homography_matrix is None:
        return None
    pt = np.array([[[px, py]]], dtype="float32")
    real_pt = cv2.perspectiveTransform(pt, inv_homography_matrix)[0][0]
    return real_pt[0], real_pt[1]

def cm_to_grid_coords(x_cm, y_cm):
    return int(x_cm // GRID_SPACING_CM), int(y_cm // GRID_SPACING_CM)

# === Grid Drawing and Homography ===

def create_and_cache_grid_overlay():
    global grid_overlay
    canvas = np.zeros((REAL_HEIGHT_CM+1, REAL_WIDTH_CM+1, 3), dtype=np.uint8)
    for x in range(0, REAL_WIDTH_CM+1, GRID_SPACING_CM):
        cv2.line(canvas, (x, 0), (x, REAL_HEIGHT_CM), GRID_LINE_COLOR, 1)
    for y in range(0, REAL_HEIGHT_CM+1, GRID_SPACING_CM):
        cv2.line(canvas, (0, y), (REAL_WIDTH_CM, y), GRID_LINE_COLOR, 1)

    w_px = 1280
    h_px = 720
    grid_overlay = cv2.warpPerspective(canvas, homography_matrix, (w_px, h_px), flags=cv2.INTER_LINEAR)
    print("🗺️ Cached grid overlay.")

def draw_metric_grid(frame):
    if grid_overlay is None:
        return frame
    blended = cv2.addWeighted(frame, 1.0, grid_overlay, 0.5, 0)
    for (gx, gy) in obstacles:
        x_cm, y_cm = gx * GRID_SPACING_CM, gy * GRID_SPACING_CM
        pt = np.array([[[x_cm, y_cm]]], dtype="float32")
        px, py = cv2.perspectiveTransform(pt, homography_matrix)[0][0]
        cv2.circle(blended, (int(px), int(py)), OBSTACLE_DRAW_RADIUS_PX, (0, 0, 255), -1)
    return blended

# === Pathfinding & Route Planning ===

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
            return path[::-1]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx <= grid_w and 0 <= ny <= grid_h and (nx, ny) not in obstacles_set:
                tentative = g_score[current] + 1
                if tentative < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative
                    f = tentative + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, (nx, ny)))
    return []

def compress_path(points):
    if len(points) < 3:
        return points[:]
    compressed = [points[0]]
    prev_dx = points[1][0] - points[0][0]
    prev_dy = points[1][1] - points[0][1]

    def norm(dx, dy):
        if dx != 0: dx = dx // abs(dx)
        if dy != 0: dy = dy // abs(dy)
        return dx, dy

    prev_dir = norm(prev_dx, prev_dy)

    for curr, nxt in zip(points[1:], points[2:]):
        dx = nxt[0] - curr[0]
        dy = nxt[1] - curr[1]
        curr_dir = norm(dx, dy)
        if curr_dir != prev_dir:
            compressed.append(curr)
        prev_dir = curr_dir
    compressed.append(points[-1])
    return compressed

def pick_top_n(balls, n=MAX_BALLS_TO_COLLECT):
    global robot_position_cm
    start_cm = robot_position_cm or START_POINT_CM
    sx, sy = cm_to_grid_coords(*start_cm)
    def dist(b):
        gx, gy = cm_to_grid_coords(b[0], b[1])
        return abs(sx - gx) + abs(sy - gy)
    return sorted(balls, key=dist)[:n]

def get_expanded_obstacles(raw):
    exp = set()
    max_g = REAL_WIDTH_CM // GRID_SPACING_CM
    max_h = REAL_HEIGHT_CM // GRID_SPACING_CM
    for (gx, gy) in raw:
        for dx in range(-BUFFER_CELLS, BUFFER_CELLS + 1):
            for dy in range(-BUFFER_CELLS, BUFFER_CELLS + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx <= max_g and 0 <= ny <= max_h:
                    exp.add((nx, ny))
    return exp

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

# === Calibration via Mouse ===

def click_to_set_corners(event, x, y, flags, param):
    global calibration_points, homography_matrix, inv_homography_matrix
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append([x, y])
            print(f"Corner {len(calibration_points)} set: ({x}, {y})")
        if len(calibration_points) == 4 and homography_matrix is None:
            dst = np.array([
                [0, 0],
                [REAL_WIDTH_CM, 0],
                [REAL_WIDTH_CM, REAL_HEIGHT_CM],
                [0, REAL_HEIGHT_CM]
            ], dtype="float32")
            src = np.array(calibration_points, dtype="float32")
            homography_matrix = cv2.getPerspectiveTransform(dst, src)
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

# === Optional Route Saving ===

def save_route_to_file(route_cm, filename="route.txt"):
    try:
        with open(filename, "w") as f:
            for x, y in route_cm:
                f.write(f"{x:.2f},{y:.2f}\n")
        print("📦 Route saved to route.txt")
    except Exception as e:
        print("❌ Error saving route:", e)

def draw_full_route(frame, ball_positions):
    global cached_route, last_ball_positions_cm, last_selected_goal
    global pending_route, full_grid_path, selected_goal

    if homography_matrix is None:
        return frame

    # Draw robot
    if robot_position_cm:
        pt = np.array([[[robot_position_cm[0], robot_position_cm[1]]]], dtype="float32")
        px, py = cv2.perspectiveTransform(pt, homography_matrix)[0][0]
        cv2.circle(frame, (int(px), int(py)), 12, (255, 0, 0), 3)
        cv2.putText(frame, "ROBOT", (int(px) - 20, int(py) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    chosen = pick_top_n(ball_positions)
    for (_, _, lbl, cx, cy) in chosen:
        cv2.circle(frame, (int(cx), int(cy)), 10, (0, 255, 0), 3)
        cv2.putText(frame, lbl, (int(cx) - 10, int(cy) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    route_changed = (
        cached_route is None or
        significant_change(chosen, last_ball_positions_cm) or
        last_selected_goal != selected_goal
    )

    if route_changed:
        if len(ball_positions) > MAX_BALLS_TO_COLLECT:
            coords = [(round(b[0], 1), round(b[1], 1), b[2]) for b in chosen]
            print(f"📋 Using only these {MAX_BALLS_TO_COLLECT} balls: {coords}")
        route_cm, grid_cells = compute_best_route(chosen, selected_goal)
        cached_route = route_cm
        last_ball_positions_cm = chosen.copy()
        last_selected_goal = selected_goal
        full_grid_path = grid_cells
    else:
        route_cm = cached_route

    pending_route = route_cm

    overlay = frame.copy()
    total_cm = 0
    grid_w = REAL_WIDTH_CM // GRID_SPACING_CM
    grid_h = REAL_HEIGHT_CM // GRID_SPACING_CM

    for i in range(len(route_cm) - 1):
        sc = cm_to_grid_coords(*route_cm[i])
        ec = cm_to_grid_coords(*route_cm[i + 1])
        path = astar(sc, ec, grid_w, grid_h, get_expanded_obstacles(obstacles))
        if not path:
            continue
        for j in range(len(path)):
            prev = path[j - 1] if j > 0 else path[0]
            curr = path[j]
            x0, y0 = prev[0] * GRID_SPACING_CM, prev[1] * GRID_SPACING_CM
            x1, y1 = curr[0] * GRID_SPACING_CM, curr[1] * GRID_SPACING_CM
            p0 = cv2.perspectiveTransform(np.array([[[x0, y0]]], dtype="float32"), homography_matrix)[0][0]
            p1 = cv2.perspectiveTransform(np.array([[[x1, y1]]], dtype="float32"), homography_matrix)[0][0]
            cv2.line(overlay, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), PATH_COLOR, 3)
            total_cm += GRID_SPACING_CM

    cv2.putText(overlay, f"Total Path: {total_cm}cm to Goal {selected_goal}",
                (10, overlay.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    return overlay


def compute_best_route(balls_list, goal_name):
    if not balls_list:
        return [], []
    start_cm = robot_position_cm or START_POINT_CM
    start_cell = cm_to_grid_coords(*start_cm)
    grid_w = REAL_WIDTH_CM // GRID_SPACING_CM
    grid_h = REAL_HEIGHT_CM // GRID_SPACING_CM
    ball_cells = [cm_to_grid_coords(b[0], b[1]) for b in balls_list]
    goal_cm = GOAL_RANGE[goal_name][0]
    goal_cell = cm_to_grid_coords(goal_cm[0], goal_cm[1])

    points = [start_cell] + ball_cells
    exp_obs = get_expanded_obstacles(obstacles)

    dm = {}
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            p, q = points[i], points[j]
            path = astar(p, q, grid_w, grid_h, exp_obs)
            dm[(i,j)] = (len(path), path)
            dm[(j,i)] = (len(path), path[::-1])

    bg = {}
    for idx, cell in enumerate(ball_cells, start=1):
        path = astar(cell, goal_cell, grid_w, grid_h, exp_obs)
        bg[idx] = (len(path), path)

    best = None
    best_cost = float('inf')
    for perm in itertools.permutations(range(1, len(ball_cells)+1)):
        cost = 0
        for a, b in zip([0]+list(perm), list(perm)+['G']):
            if b == 'G':
                c, _ = bg[a]
            else:
                c, _ = dm[(a,b)]
            cost += c
        if cost < best_cost:
            best_cost, best = cost, perm

    route_cm = [start_cm] + [(balls_list[i-1][0], balls_list[i-1][1]) for i in best] + [goal_cm]
    full_cells = []
    for a, b in zip([0]+list(best), list(best)+['G']):
        if b == 'G':
            _, path = bg[a]
        else:
            _, path = dm[(a,b)]
        full_cells.extend(path)

    return route_cm, full_cells