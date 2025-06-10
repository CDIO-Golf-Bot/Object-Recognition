import cv2
import numpy as np

class GridManager:
    def __init__(self, config):
        self.real_w = config.REAL_WIDTH_CM
        self.real_h = config.REAL_HEIGHT_CM
        self.spacing = config.GRID_SPACING_CM
        self.cal_points = []
        self.h_matrix = None
        self.obstacles = set()
        self.ignored = config.IGNORED_AREA
        self.start_cm = config.START_POINT_CM
        self.goals = config.GOAL_RANGE

    def add_calibration_point(self, pt):
        if len(self.cal_points) < config.CALIBRATION_POINTS_REQUIRED:
            self.cal_points.append(pt)
        if len(self.cal_points) == config.CALIBRATION_POINTS_REQUIRED:
            self._compute_homography()

    def _compute_homography(self):
        dst = np.array([
            [0, 0],
            [self.real_w, 0],
            [self.real_w, self.real_h],
            [0, self.real_h]
        ], dtype='float32')
        src = np.array(self.cal_points, dtype='float32')
        self.h_matrix = cv2.getPerspectiveTransform(dst, src)
        self._mark_center_obstacle()
        self._clear_outer_edges()

    def pixel_to_cm(self, x, y):
        if self.h_matrix is None:
            return None
        pt = np.array([[[x, y]]], dtype='float32')
        inv = np.linalg.inv(self.h_matrix)
        real = cv2.perspectiveTransform(pt, inv)[0][0]
        x_cm, y_cm = real
        return x_cm, self.real_h - y_cm

    def cm_to_grid(self, x_cm, y_cm):
        return int(x_cm // self.spacing), int(y_cm // self.spacing)

    def toggle_obstacle(self, gx, gy):
        if (gx, gy) in self.obstacles:
            self.obstacles.remove((gx, gy))
        else:
            self.obstacles.add((gx, gy))

    def _mark_center_obstacle(self):
        cx, cy = self.real_w / 2, self.real_h / 2
        half = 10
        for x in range(int(cx - half), int(cx + half), self.spacing):
            for y in range(int(cy - half), int(cy + half), self.spacing):
                self.obstacles.add(self.cm_to_grid(x, y))

    def _clear_outer_edges(self):
        max_x = self.real_w // self.spacing
        max_y = self.real_h // self.spacing
        for i in range(max_x + 1):
            self.obstacles.discard((i, 0))
            self.obstacles.discard((i, max_y))
        for j in range(max_y + 1):
            self.obstacles.discard((0, j))
            self.obstacles.discard((max_x, j))