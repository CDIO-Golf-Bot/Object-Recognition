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

    def get_corners(self):
        return self.topLeft, self.topRight, self.botLeft, self.botRight

    def get_dimensions(self):
        """Returns the dimensions of the maze."""
        if self.topLeft and self.botRight:
            width = abs(self.topRight.x - self.topLeft.x)
            height = abs(self.topRight.y - self.botRight.y)
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

    def draw_grid_and_coordinates(self, frame, max_cell_size):
        """Draw grid lines and display coordinates on the maze."""
        if not self.topLeft or not self.botRight:
            return  # Do nothing if corners are not set yet

        # Get maze's width and height
        maze_width, maze_height = self.get_dimensions()

        # Calculate ideal grid size
        grid_width = maze_width // max_cell_size  # Number of columns
        grid_height = maze_height // max_cell_size  # Number of rows

        # Calculate step size for each grid cell
        x_step = maze_width / grid_width
        y_step = maze_height / grid_height

        font_scale = 0.3
        font_color = (255, 0, 0)
        font_thickness = 1

        for i in range(grid_width + 1):  # horizontal grid
            for j in range(grid_height + 1):  # vertical grid
                x = int(self.topLeft.x + i * x_step)
                y = int(self.topLeft.y + j * y_step)

                # Draw vertical grid lines
                if i < grid_width + 1:
                    cv2.line(frame, (x, self.topLeft.y), (x, self.botLeft.y), (0, 255, 0), 1)

                # Draw horizontal grid lines
                if j < grid_height + 1:
                    cv2.line(frame, (self.topLeft.x, y), (self.topRight.x, y), (0, 255, 0), 1)

                # Display grid coordinates for each cell
                grid_coords = self.pixel_to_grid(x, y, grid_size=(grid_width, grid_height))
                text_offset_x = 3
                text_offset_y = -3
                cv2.putText(frame, f'{grid_coords[0]},{grid_coords[1]}',
                            (x + text_offset_x, y + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            font_color, font_thickness, cv2.LINE_AA)

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

    initial_size = 30
    cv2.createTrackbar("Cell Size", "Maze Setup with Grid", initial_size, 100, lambda x: None)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        size_val = max(cv2.getTrackbarPos("Cell Size", "Maze Setup with Grid"), 1)
        maze.draw_grid_and_coordinates(frame, max_cell_size=size_val)

        # Show the frame with the maze and grid coordinates
        cv2.imshow("Maze Setup with Grid", frame)

        # Exit the loop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
