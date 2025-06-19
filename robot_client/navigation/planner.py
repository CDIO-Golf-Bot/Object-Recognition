"""
planner.py

Implement path planning and route visualization on the grid:
  • A* search, Manhattan heuristic, and path compression.
  • Select top N balls and brute-force TSP-style best route to a goal.
  • Compute and cache full route in cm & grid cells.
  • Draw the complete path with real-world scaling on camera frames.
  • Save a planned route to disk.
"""


import cv2
import numpy as np
import itertools
import heapq

from robot_client import config
from .. import utils
from . import grid_utils as gu
from . import calibration as cal

# === Routing State ===
cached_route = None
last_ball_positions_cm = []
last_selected_goal = None
full_grid_path = []
pending_route = None
selected_goal = 'A'
robot_position_cm = None


def save_route_to_file(route_cm, filename="route.txt"):
    """
    Save a list of (x_cm,y_cm) tuples to a text file.
    """
    try:
        with open(filename, "w") as f:
            for x, y in route_cm:
                f.write(f"{x:.2f},{y:.2f}\n")
        print("📦 Route saved to route.txt")
    except Exception as e:
        print(f"❌ Error saving route: {e}")


def heuristic(a, b):
    """Manhattan distance heuristic for grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, grid_w, grid_h, obstacles_set):
    """
    A* search from start to goal on a grid.
    start/goal are (gx,gy) cell coords.
    """
    open_set, came_from, g_score = [(0, start)], {}, {start: 0}
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
                    came_from[(nx, ny)], g_score[(nx, ny)] = current, tentative
                    f = tentative + heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f, (nx, ny)))
    return []


def compress_path(points):
    """
    Remove intermediate colinear points in a path of (x_cm, y_cm).
    """
    if len(points) < 3:
        return points[:]
    compressed = [points[0]]
    def norm(dx, dy):
        length = (dx**2 + dy**2)**0.5
        if length == 0: return (0, 0)
        return (dx/length, dy/length)
    prev_dir = norm(points[1][0]-points[0][0], points[1][1]-points[0][1])
    for curr, nxt in zip(points[1:], points[2:]):
        dir = norm(nxt[0]-curr[0], nxt[1]-curr[1])
        if abs(dir[0] - prev_dir[0]) > 1e-3 or abs(dir[1] - prev_dir[1]) > 1e-3:
            compressed.append(curr)
            prev_dir = dir
    compressed.append(points[-1])
    return compressed


def pick_top_n(balls, n=None):
    """
    Choose the n closest balls from robot_position_cm.
    """
    global robot_position_cm
    if n is None:
        n = config.MAX_BALLS_TO_COLLECT
    start = robot_position_cm or config.START_POINT_CM
    sx, sy = gu.cm_to_grid_coords(*start)
    def dist(b):
        gx, gy = gu.cm_to_grid_coords(b[0], b[1])
        return abs(sx - gx) + abs(sy - gy)
    return sorted(balls, key=dist)[:n]
    


def get_expanded_obstacles(raw):
    """
    Expand raw obstacle seeds by BUFFER_CELLS.
    """
    exp = set()
    max_g = config.REAL_WIDTH_CM // config.GRID_SPACING_CM
    max_h = config.REAL_HEIGHT_CM // config.GRID_SPACING_CM
    for gx, gy in raw:
        for dx in range(-config.BUFFER_CELLS, config.BUFFER_CELLS + 1):
            for dy in range(-config.BUFFER_CELLS, config.BUFFER_CELLS + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx <= max_g and 0 <= ny <= max_h:
                    exp.add((nx, ny))
    return exp

def compute_best_route(balls_list, goal_name):
    """
    TSP-like brute-force over up to MAX_BALLS: finds shortest route through balls to goal.
    Returns (route_cm_list, full_route_cells).
    Only plans if robot_position_cm is known; otherwise returns empty lists.
    """
    global full_grid_path, pending_route, cached_route, last_ball_positions_cm, last_selected_goal

    # 1️⃣ Bail out if we don't have a valid robot start pose
    if robot_position_cm is None:
        print("⚠️  Cannot plan route: robot position unknown (no ArUco).")
        return [], []

    # 2️⃣ No balls detected: nothing to do
    if not balls_list:
        return [], []
    
    # Limit top N balls to MAX_BALLS_TO_COLLECT
    balls_list = pick_top_n(balls_list)

    # 3️⃣ Use only the ArUco-derived starting position
    start_cm = robot_position_cm
    start_cell = gu.cm_to_grid_coords(*start_cm)
    grid_w = config.REAL_WIDTH_CM // config.GRID_SPACING_CM
    grid_h = config.REAL_HEIGHT_CM // config.GRID_SPACING_CM
    ball_cells = [gu.cm_to_grid_coords(b[0], b[1]) for b in balls_list]
    goal_cm = config.GOAL_RANGE[goal_name][0]
    goal_cell = gu.cm_to_grid_coords(goal_cm[0], goal_cm[1])

    # Build distance map between start + balls
    dm = {}
    points = [start_cell] + ball_cells
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            path = astar(points[i], points[j], grid_w, grid_h, gu.obstacles)
            dm[(i, j)] = (len(path), path)
            dm[(j, i)] = (len(path), list(reversed(path)))

    # Distances from each ball to goal
    bg = {}
    for idx, cell in enumerate(ball_cells, start=1):
        path_to_goal = astar(cell, goal_cell, grid_w, grid_h, gu.obstacles)
        bg[idx] = (len(path_to_goal), path_to_goal)

    # Permute through ball pickup orders
    best_perm, best_cost = None, float('inf')
    indices = list(range(1, len(ball_cells) + 1))
    for perm in itertools.permutations(indices):
        cost = 0
        seq = [0] + list(perm)
        for a, b in zip(seq, list(perm) + ['G']):
            cost += bg[a][0] if b == 'G' else dm[(a, b)][0]
        if cost < best_cost:
            best_cost, best_perm = cost, perm

    # Build route in cm and grid cells
    route_cm = [start_cm] + [(balls_list[i-1][0], balls_list[i-1][1]) for i in best_perm] + [goal_cm]
    full_cells = []
    for a, b in zip([0] + list(best_perm), list(best_perm) + ['G']):
        segment = bg[a][1] if b == 'G' else dm[(a, b)][1]
        full_cells.extend(segment)

    # Cache and return
    pending_route = route_cm
    cached_route = route_cm
    last_selected_goal = goal_name
    last_ball_positions_cm = balls_list.copy()
    full_grid_path = full_cells
    return route_cm[1:], full_cells     # skip first element (start)

def draw_full_route(frame, ball_positions):
    """
    Draw the route (grid_lines + robot + path) on the given frame,
    with a small dot marking the robot's starting position.
    """
    if cal.homography_matrix is None:
        return frame

    # Draw robot start as a 2px filled dot for precision
    if robot_position_cm:
        pt = np.array([[[robot_position_cm[0], robot_position_cm[1]]]], dtype="float32")
        px, py = cv2.perspectiveTransform(pt, cal.homography_matrix)[0][0]
        px_i, py_i = int(round(px)), int(round(py))
        # Red in BGR, radius=4 for prominence
        cv2.circle(frame, (px_i, py_i), 4, (0, 0, 255), -1)
        cv2.putText(frame, "ROBOT", (px_i - 20, py_i - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # (rest of the overlay drawing logic remains unchanged)
    
    # Recompute if needed
    global cached_route, full_grid_path
    changed = (cached_route is None or
               utils.significant_change(ball_positions, last_ball_positions_cm) or
               last_selected_goal != selected_goal)
    if changed:
        route_cm, grid_cells = compute_best_route(ball_positions, selected_goal)
        cached_route, full_grid_path = route_cm, grid_cells
    else:
        route_cm = cached_route

    # Overlay path
    overlay = frame.copy()
    total_cm = 0
    grid_w = config.REAL_WIDTH_CM // config.GRID_SPACING_CM
    grid_h = config.REAL_HEIGHT_CM // config.GRID_SPACING_CM
    for p1, p2 in zip(route_cm[:-1], route_cm[1:]):
        sc = gu.cm_to_grid_coords(*p1)
        ec = gu.cm_to_grid_coords(*p2)
        path = astar(sc, ec, grid_w, grid_h, gu.obstacles)
        for prev, curr in zip(path, path[1:]):
            x0, y0 = prev[0]*config.GRID_SPACING_CM, prev[1]*config.GRID_SPACING_CM
            x1, y1 = curr[0]*config.GRID_SPACING_CM, curr[1]*config.GRID_SPACING_CM
            p0 = cv2.perspectiveTransform(np.array([[[x0, y0]]], dtype="float32"), cal.homography_matrix)[0][0]
            p1 = cv2.perspectiveTransform(np.array([[[x1, y1]]], dtype="float32"), cal.homography_matrix)[0][0]
            cv2.line(overlay, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), config.PATH_COLOR, 3)
            total_cm += config.GRID_SPACING_CM

    cv2.putText(overlay, f"Total Path: {total_cm}cm to Goal {selected_goal}", (10, overlay.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.TEXT_COLOR, 2)
    return overlay
