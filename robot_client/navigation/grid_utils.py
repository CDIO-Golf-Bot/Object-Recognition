"""
grid_utils.py

Provide real-world ↔ grid coordinate services and obstacle drawing:
  • Convert pixel ⇄ cm (pixel_to_cm / cm_to_grid_coords).
  • Maintain an obstacle set in grid coordinates.
  • Overlay the cached metric grid and draw obstacles onto each frame.
  • Utility to clear border cells (ensure_outer_edges_walkable).
"""

import cv2
import numpy as np
from robot_client import config
from . import calibration

# === Grid & Obstacle State ===
obstacles = set()

def pixel_to_cm(px, py):
    """
    Map pixel coordinates to real-world cm using calibration inverse homography.
    Returns (x_cm, y_cm) or None if homography not set.
    """
    if calibration.inv_homography_matrix is None:
        return None
    pt = np.array([[[px, py]]], dtype="float32")
    real_pt = cv2.perspectiveTransform(pt, calibration.inv_homography_matrix)[0][0]
    return real_pt[0], real_pt[1]

def cm_to_grid_coords(x_cm, y_cm):
    """
    Convert real-world cm position to grid cell indices.
    """
    return int(x_cm // config.GRID_SPACING_CM), int(y_cm // config.GRID_SPACING_CM)

def get_border_buffer_obstacles():
    """Generate all edge cells within BORDER_BUFFER_CELLS—but carve out around goals."""
    max_g = config.REAL_WIDTH_CM // config.GRID_SPACING_CM
    max_h = config.REAL_HEIGHT_CM // config.GRID_SPACING_CM
    border_obs = set()
    # top/bottom bands
    for gx in range(max_g+1):
        for gy in range(config.BORDER_BUFFER_CELLS):
            border_obs.add((gx, gy))
        for gy in range(max_h - config.BORDER_BUFFER_CELLS + 1, max_h+1):
            border_obs.add((gx, gy))
    # left/right bands
    for gy in range(max_h+1):
        for gx in range(config.BORDER_BUFFER_CELLS):
            border_obs.add((gx, gy))
        for gx in range(max_g - config.BORDER_BUFFER_CELLS + 1, max_g+1):
            border_obs.add((gx, gy))
    # carve out goal zones
    for goals in config.GOAL_RANGE.values():
        if not goals: continue
        for goal_cm in goals:
            gx0, gy0 = cm_to_grid_coords(goal_cm[0], goal_cm[1])
            for dx in range(-config.BORDER_BUFFER_CELLS, config.BORDER_BUFFER_CELLS+1):
                for dy in range(-config.BORDER_BUFFER_CELLS, config.BORDER_BUFFER_CELLS+1):
                    border_obs.discard((gx0+dx, gy0+dy))
    

def draw_metric_grid(frame):
    """
    Overlay the cached grid and draw obstacle points.
    """
    if calibration.grid_overlay is None:
        return frame
    # blend grid
    blended = cv2.addWeighted(frame, 1.0, calibration.grid_overlay, 0.5, 0)
    # draw obstacles
    for gx, gy in obstacles:
        x_cm, y_cm = gx * config.GRID_SPACING_CM, gy * config.GRID_SPACING_CM
        pt = np.array([[[x_cm, y_cm]]], dtype="float32")
        px, py = cv2.perspectiveTransform(pt, calibration.homography_matrix)[0][0]
        cv2.circle(blended, (int(px), int(py)), config.OBSTACLE_DRAW_RADIUS_PX, (0, 0, 255), -1)
    return blended

def ensure_outer_edges_walkable():
    """
    Clear any obstacles on the border cells of the grid.
    """
    max_x = config.REAL_WIDTH_CM // config.GRID_SPACING_CM
    max_y = config.REAL_HEIGHT_CM // config.GRID_SPACING_CM
    for gx in range(max_x + 1):
        obstacles.discard((gx, 0))
        obstacles.discard((gx, max_y))
    for gy in range(max_y + 1):
        obstacles.discard((0, gy))
        obstacles.discard((max_x, gy))
    print("✅ Outer edges cleared.")

def toggle_obstacle_at_pixel(x, y):
    if calibration.inv_homography_matrix is None: return
    cm = pixel_to_cm(x, y)
    if not cm: return
    gx, gy = cm_to_grid_coords(*cm)
    if (gx, gy) in obstacles: obstacles.remove((gx,gy))
    else:                  obstacles.add((gx,gy))


def world_to_centered(x_cm, y_cm):
    """
    Convert (0,0)=top-left → (0,0)=table-center.
    """
    cx = x_cm - config.REAL_WIDTH_CM  / 2.0
    cy = y_cm - config.REAL_HEIGHT_CM / 2.0
    return cx, cy
