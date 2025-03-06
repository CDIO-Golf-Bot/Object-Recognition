import cv2
import numpy as np
from navigation.Point import Point

class Maze:
    def __init__(self, topLeft=None, topRight=None, botLeft=None, botRight=None):
        self.topLeft = topLeft
        self.topRight = topRight
        self.botLeft = botLeft
        self.botRight = botRight
        self.top_left_clicked = False
        self.bottom_right_clicked = False
        self.grid_points = {}  # Dictionary to store grid points

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

    def pixel_to_grid(self, x, y, grid_size=(20, 20)):
        """Map pixel coordinates to grid coordinates."""
        maze_width, maze_height = self.get_dimensions()
        x_scale = maze_width / grid_size[0]
        y_scale = maze_height / grid_size[1]

        grid_x = int((x - self.topLeft.x) / x_scale)
        grid_y = int((y - self.topLeft.y) / y_scale)

        return grid_x, grid_y

    def grid_to_pixel(self, grid_x, grid_y, grid_size=(20, 20)):
        """Convert grid coordinates to pixel coordinates."""
        if not self.topLeft:  # Ensure that topLeft is initialized
            print("Error: Maze corners not set yet.")
            return 0, 0  # Return default values when corners are not set
        
        maze_width, maze_height = self.get_dimensions()
        x_scale = maze_width / grid_size[0]
        y_scale = maze_height / grid_size[1]

        pixel_x = self.topLeft.x + grid_x * x_scale
        pixel_y = self.topLeft.y + grid_y * y_scale

        return int(pixel_x), int(pixel_y)

    def draw_grid_and_coordinates(self, frame, grid_size=(20, 20)):
        """Draw grid lines and display coordinates at the intersections."""
        if not self.topLeft or not self.botRight:
            return  # Do nothing if corners are not set yet

        # Get maze's width and height
        maze_width, maze_height = self.get_dimensions()

        # Calculate ideal grid size based on hardcoded grid_size
        grid_width = grid_size[0]  # Number of columns (hardcoded)
        grid_height = grid_size[1]  # Number of rows (hardcoded)

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

                # Draw vertical grid lines
                if i < grid_width + 1:
                    cv2.line(frame, (x, self.topLeft.y), (x, self.botLeft.y), (0, 255, 0), 1)

                # Draw horizontal grid lines
                if j < grid_height + 1:
                    cv2.line(frame, (self.topLeft.x, y), (self.topRight.x, y), (0, 255, 0), 1)

                # Display grid coordinates at the intersection points
                grid_coords = (i, j)  # Grid coordinates are simply (i, j)
                text_offset_x = 5  # Adjust text position for better visibility
                text_offset_y = -5
                cv2.putText(frame, f'{grid_coords[0]},{grid_coords[1]}',
                            (x + text_offset_x, y + text_offset_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
                    
                    
    def get_point(self, grid_x, grid_y):
        return self.grid_points.get((grid_x, grid_y))

    def set_point(self, grid_x, grid_y, point):
        self.grid_points[(grid_x, grid_y)] = point

# Mouse callback to set corners of the maze
def mouse_callback(event, x, y, flags, param):
    """Mouse callback to set corners of the maze."""
    maze = param
    if event == cv2.EVENT_LBUTTONDOWN:
        maze.set_corners(x, y)

def main():
    """Main function to run the maze setup with camera feed."""
    maze = Maze()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Camera resolution for maze
    camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set mouse callback for user input
    cv2.namedWindow("Maze Setup with Grid")
    cv2.setMouseCallback("Maze Setup with Grid", mouse_callback, maze)

    # Set a hardcoded grid size
    hardcoded_grid_size = (20, 20)  # Set the number of rows and columns for the grid

    # Flag to check if the maze is initialized
    maze_initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        maze.draw_grid_and_coordinates(frame, grid_size=hardcoded_grid_size)

        # Only print grid points once the maze corners are set (initialized)
        if maze.topLeft and maze.botRight and not maze_initialized:
            maze_initialized = True
            maze.print_grid_points()  # Print all grid points

        # Show the frame with the maze and grid coordinates
        cv2.imshow("Maze Setup with Grid", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()