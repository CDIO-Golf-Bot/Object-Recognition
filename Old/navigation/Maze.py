import cv2
import numpy as np
from navigation.Point import Point
from collections import deque

class Maze:
    def __init__(self, topLeft=None, topRight=None, botLeft=None, botRight=None, gridSize=(10, 10)):  # can change gridSize
        self.topLeft = topLeft
        self.topRight = topRight
        self.botLeft = botLeft
        self.botRight = botRight
        self.top_left_clicked = False
        self.bottom_right_clicked = False
        self.gridSize = gridSize
        self.grid_points = {}  # Dictionary to store grid points
        self.grid2D = np.zeros((gridSize[0], gridSize[1]), dtype=int)

    def get_grid(self):
        """Return the maze grid (2D array) for pathfinding."""
        return self.grid2D
    
    def get_corners(self):
        return self.topLeft, self.topRight, self.botLeft, self.botRight

    def print_grid_points(self):
        """Print all the grid points."""
        for (grid_x, grid_y), point in self.grid_points.items():
            print(f"Grid ({grid_x}, {grid_y}) has pixel point ({point.x}, {point.y})")

    def get_dimensions(self):
        """Returns the dimensions of the maze."""
        if self.topLeft and self.botRight:
            width = abs(self.botRight.x - self.topLeft.x)
            height = abs(self.botRight.y - self.topLeft.y)
            return width, height
        return 0, 0

    def set_corners(self, x, y):
        """Set the maze corners based on mouse click."""
        if not self.top_left_clicked:  # First click: top left corner
            self.topLeft = Point(x, y)
            self.top_left_clicked = True
            print(f"Top Left: {self.topLeft}")
        elif not self.bottom_right_clicked:  # Second click: bottom-right corner
            self.botRight = Point(x, y)
            self.topRight = Point(x, self.topLeft.y)  # Same y as topLeft for topRight
            self.botLeft = Point(self.topLeft.x, y)  # Same x as topLeft for botLeft
            self.bottom_right_clicked = True
            print(f"Bottom Right: {self.botRight}")

    def pixel_to_grid(self, x, y):
        """Map pixel coordinates to grid coordinates."""
        maze_width, maze_height = self.get_dimensions()
        x_scale = maze_width / self.gridSize[0]
        y_scale = maze_height / self.gridSize[1]

        grid_x = int((x - self.topLeft.x) / x_scale)
        grid_y = int((y - self.topLeft.y) / y_scale)

        return grid_x, grid_y

    def grid_to_pixel(self, grid_x, grid_y):
        """Convert grid coordinates to pixel coordinates."""
        if not self.topLeft:  # Ensure that topLeft is initialized
            print("Error: Maze corners not set yet.")
            return 0, 0  # Return default values when corners are not set
        
        maze_width, maze_height = self.get_dimensions()
        x_scale = maze_width / self.gridSize[0]
        y_scale = maze_height / self.gridSize[1]

        pixel_x = self.topLeft.x + grid_x * x_scale
        pixel_y = self.topLeft.y + grid_y * y_scale
        return int(pixel_x), int(pixel_y)

    def draw_grid_and_coordinates(self, frame):
        """Draw grid lines and display coordinates at the intersections."""
        if not self.topLeft or not self.botRight:
            return  # Do nothing if corners are not set yet

        # Get maze's width and height
        maze_width, maze_height = self.get_dimensions()

        grid_width = self.gridSize[0]
        grid_height = self.gridSize[1] 

        # Calculate step size for each grid cell
        x_step = maze_width / grid_width
        y_step = maze_height / grid_height

        font_scale = 0.3
        font_color = (255, 0, 0)  # Red color for text
        font_thickness = 1

        for i in range(grid_width + 1):  # horizontal grid
            for j in range(grid_height + 1):  # vertical grid
                # Calculate the pixel position based on the grid index
                x = int(self.topLeft.x + i * x_step)
                y = int(self.topLeft.y + j * y_step)

                cv2.circle(frame, (x, y), 3, (0, 0, 225), 1)

                # Draw vertical grid lines
                if i < grid_width + 1:
                    cv2.line(frame, (x, self.topLeft.y), (x, self.botLeft.y), (0, 255, 0), 1)

                # Draw horizontal grid lines
                if j < grid_height + 1:
                    cv2.line(frame, (self.topLeft.x, y), (self.topRight.x, y), (0, 255, 0), 1)

                # Display grid coordinates at the intersection points
                grid_coords = (i, j) 
                text_offset_x = 5 # text position
                text_offset_y = -5
                cv2.putText(frame, f'{grid_coords[0]},{grid_coords[1]}',
                            (x + text_offset_x, y + text_offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
                    
                    
    def get_point(self, grid_x, grid_y):
        """Get the pixel coordinates for a given grid point."""
        return self.grid_to_pixel(grid_x, grid_y)

    def set_point(self, grid_x, grid_y, point):
        self.grid_points[(grid_x, grid_y)] = point

    def bfs(self, start, end):
        """Find the shortest path from start to end using BFS."""
        rows, cols = self.gridSize
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        queue = deque([(start, [start])])  # Queue of (current position, path)
        visited = set([start])

        while queue:
            (current, path) = queue.popleft()
            if current == end:
                return path  # Return the path if the end is reached

            for direction in directions:
                next_row = current[0] + direction[0]
                next_col = current[1] + direction[1]
                next_pos = (next_row, next_col)

                # Check if the next position is within bounds and not blocked
                if (0 <= next_row < rows and 0 <= next_col < cols and
                    self.grid2D[next_row][next_col] == 0 and next_pos not in visited):
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))

        return None  # No path found

    def draw_path(self, frame, path):
        """Draw the path on the frame."""
        if not path:
            print("Error: Path is empty or invalid.")
            return

        for i in range(len(path) - 1):
            start = self.grid_to_pixel(path[i][0], path[i][1])
            end = self.grid_to_pixel(path[i + 1][0], path[i + 1][1])

            # Draw the line
            cv2.line(frame, start, end, (255, 255, ), 4)  # Draw a blue line for the path

    
# Mouse callback to set corners of the maze
def mouse_callback(event, x, y, flags, param):
    """Mouse callback to set corners of the maze."""
    maze = param
    if event == cv2.EVENT_LBUTTONDOWN:
        maze.set_corners(x, y)