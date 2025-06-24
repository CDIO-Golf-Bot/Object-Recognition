"""
planner.py

Implement path planning and route visualization on the grid:
  • A* search, Manhattan heuristic, and path compression.
  • Select top N balls and brute-force TSP-style best route to a goal.
  • Compute and cache full route in cm & grid cells.
  • Draw the complete path with real-world scaling on camera frames.
  • Save a planned route to disk.
"""


import math
import threading
import time
import cv2
import numpy as np
import itertools
import heapq

from robot_client import config, navigation, robot_comm, detection
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

dynamic_route = True        # starts camera reconition and route planning
route_active = False        # routes keeps running
route_enabled = False       # first start of route when pressing 's'

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

def cm_to_grid_coords(x_cm, y_cm):
    return int(x_cm // config.GRID_SPACING_CM), int(y_cm // config.GRID_SPACING_CM)

def compress_path(points):
    """
    Remove intermediate colinear points in a grid path.
    """
    if len(points) < 3:
        return points[:]
    compressed = [points[0]]
    def norm(dx, dy):
        return (dx // abs(dx) if dx else 0, dy // abs(dy) if dy else 0)
    prev_dir = norm(points[1][0]-points[0][0], points[1][1]-points[0][1])
    for curr, nxt in zip(points[1:], points[2:]):
        curr_dir = norm(nxt[0]-curr[0], nxt[1]-curr[1])
        if curr_dir != prev_dir:
            compressed.append(curr)
        prev_dir = curr_dir
    compressed.append(points[-1])
    return compressed


def pick_top_n(balls, n=None):
    """
    Choose the n closest balls from robot_position_cm, ignoring balls in the wall buffer zone
    and any obstacle (including the red cross).
    """
    global robot_position_cm

    if n is None:
        n = config.MAX_BALLS_TO_COLLECT
    start = robot_position_cm or config.START_POINT_CM
    sx, sy = gu.cm_to_grid_coords(*start)

    # Get buffer cells
    buffer_cells = gu.get_border_buffer_obstacles()

    def in_buffer(b):
        gx, gy = gu.cm_to_grid_coords(b[0], b[1])
        return (gx, gy) in buffer_cells

    def in_obstacle(b):
        gx, gy = gu.cm_to_grid_coords(b[0], b[1])
        return (gx, gy) in gu.obstacles

    def dist(b):
        gx, gy = gu.cm_to_grid_coords(b[0], b[1])
        return abs(sx - gx) + abs(sy - gy)

    # Filter out balls in the buffer zone or in any obstacle, then sort and pick top n
    filtered_balls = [b for b in balls if not in_buffer(b) and not in_obstacle(b)]
    return sorted(filtered_balls, key=dist)[:n]

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



def greedy_route(points, distance_map):
    """
    Greedily find a route through points using precomputed distances.
    Returns a list of indices representing the route.
    """
    unvisited = set(range(1, len(points)))  # skip start (0)
    route = [0]
    curr = 0
    while unvisited:
        next_node = min(unvisited, key=lambda j: distance_map[(curr, j)][0])
        route.append(next_node)
        unvisited.remove(next_node)
        curr = next_node
    return route

def compute_axis_aligned_approach_point(from_pt, to_pt, approach_dist=20):
    """
    Return a point approach_dist cm before to_pt, but only along the axis (x or y)
    that has the largest difference, so the robot always approaches in a straight line.
    """
    dx = to_pt[0] - from_pt[0]
    dy = to_pt[1] - from_pt[1]
    if abs(dx) > abs(dy):
        # Approach along x axis
        if abs(dx) < approach_dist:
            return from_pt
        x_approach = to_pt[0] - approach_dist * (1 if dx > 0 else -1)
        y_approach = to_pt[1]
    else:
        # Approach along y axis
        if abs(dy) < approach_dist:
            return from_pt
        x_approach = to_pt[0]
        y_approach = to_pt[1] - approach_dist * (1 if dy > 0 else -1)
    return (x_approach, y_approach)


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

    gu.obstacles |= gu.get_border_buffer_obstacles()

    # Build distance map between start + balls
    dm = {}
    points = [start_cell] + ball_cells
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            path = astar(points[i], points[j], grid_w, grid_h, gu.obstacles)
            dm[(i, j)] = (len(path), path)
            dm[(j, i)] = (len(path), list(reversed(path)))

     # Use greedy_route to get the order
    route_indices = greedy_route(points, dm)

   # Build route in cm with approach points before each ball
    route_cm = [start_cm] 
    for idx in range(1, len(route_indices) - 1):  # skip start and goal for approach
        curr_idx = route_indices[idx]
        prev_pt = route_cm[-1]
        ball_pt = balls_list[curr_idx - 1][0:2]  # balls_list is offset by 1
        # Insert approach point before the ball
        approach_pt = compute_axis_aligned_approach_point(prev_pt, ball_pt, approach_dist=20)    
        if approach_pt != prev_pt:
            route_cm.append(approach_pt)
        route_cm.append(ball_pt)
    goal_pt = goal_cm
    prev_pt = route_cm[-1]
    approach_pt = compute_axis_aligned_approach_point(prev_pt, goal_pt, approach_dist=0)
    if approach_pt != prev_pt:
        route_cm.append(approach_pt)
    route_cm.append(goal_pt)
    if approach_pt != prev_pt:
            route_cm.append(approach_pt)
    route_cm.append(goal_pt)

    # Build full grid cell path for visualization
    full_cells = []
    for a, b in zip(route_indices, route_indices[1:]):
        full_cells.extend(dm[(a, b)][1])
    # Add path from last ball to goal
    last_ball_idx = route_indices[-1]
    path_to_goal = astar(points[last_ball_idx], goal_cell, grid_w, grid_h, gu.obstacles)
    full_cells.extend(path_to_goal)

    # Cache and return
    pending_route = route_cm
    cached_route = route_cm
    last_selected_goal = goal_name
    last_ball_positions_cm = balls_list.copy()
    full_grid_path = full_cells
    return route_cm[0:], full_cells

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
    
    # Recompute if needed
    global cached_route, full_grid_path
    changed = (cached_route is None or
               utils.significant_change(ball_positions, last_ball_positions_cm) or
               last_selected_goal != selected_goal)
    if dynamic_route and changed:
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


def plan_route():
    """Plan a route and return waypoints (in cm) and grid cells."""
    balls = detection.ball_positions_cm
    if not balls:
        print("No balls detected; cannot plan route.")
        return None, None
    route_cm, grid_cells = compute_best_route(
        balls,
        navigation.selected_goal
    )
    if route_cm:
        turn_points = navigation.compress_path(grid_cells)
        turn_points_cm = navigation.grid_path_to_cm(turn_points)
        navigation.pending_route = turn_points_cm
        navigation.full_grid_path = grid_cells
        print(f"Planned route: {len(turn_points_cm)} waypoints")
        return turn_points_cm, grid_cells
    else:
        print("Route computation returned no waypoints.")
        return None, None

def execute_route(route_cm, stop_event):
    """Drive each segment until arrival, with abort & timeout."""
    global route_active
    route_active = True
    for x_target, y_target in route_cm:
        if stop_event.is_set():
            print("Route aborted by user")
            return
        robot_comm.send_goto(x_target, y_target)
        start_t = time.time()
        while True:
            if stop_event.is_set():
                print("Route aborted during segment")
                return
            elapsed = time.time() - start_t
            if elapsed > config.MAX_SEGMENT_TIME:
                print(f"⚠️ Timeout ({elapsed:.1f}s) to reach ({x_target:.1f},{y_target:.1f}); skipping")
                break
            pos = detection.planner.robot_position_cm
            if pos is not None:
                cur_x, cur_y = pos
                dist = math.hypot(x_target - cur_x, y_target - cur_y)
                if dist < config.ARRIVAL_THRESHOLD_CM:
                    print(f"Arrived at ({x_target:.1f}, {y_target:.1f}) → d={dist:.1f}cm in {elapsed:.1f}s")
                    break
            time.sleep(0.05)
    robot_comm.send_deliver()
    print("🏁 Route complete, delivered")
    route_active = False

def plan_and_execute(stop_event):
    """Plan a route and execute it in a new thread if possible."""
    route_cm, _ = plan_route()  # Ignore grid_cells if not used
    if route_cm:
        t = threading.Thread(
            target=execute_route,
            args=(route_cm, stop_event),
            daemon=True
        )
        t.start()
        return t
    else:
        print("No route to execute.")
        return None