import cv2
import numpy as np
from points import Point  # Assuming Point class is in points.py
class Maze:
    def __init__(self, topLeft=None, topRight=None, botLeft=None, botRight=None):
        self.topLeft = topLeft
        self.topRight = topRight
        self.botLeft = botLeft
        self.botRight = botRight

    def get_corners(self):
        return self.topLeft, self.topRight, self.botLeft, self.botRight

    def get_dimensions(self):
        if self.topLeft and self.botRight:
            width = abs(self.topRight.x - self.topLeft.x)
            height = abs(self.topRight.y - self.botRight.y)
            return width, height
        return 0, 0

# Function to map pixel coordinates to grid coordinates
def pixel_to_grid(x, y, maze: Maze, grid_size=(20, 20)):
    maze_width, maze_height = maze.get_dimensions()
    x_scale = maze_width / grid_size[0]
    y_scale = maze_height / grid_size[1]

    grid_x = int((x - maze.topLeft.x) / x_scale)
    grid_y = int((y - maze.topLeft.y) / y_scale)

    return grid_x, grid_y

# Function to draw grid lines and coordinates
def draw_grid_and_coordinates(frame, maze: Maze, max_cell_size):
    if not maze.topLeft or not maze.botRight:
        return  # Do nothing if corners are not set yet

    # Get the maze's width and height
    maze_width, maze_height = maze.get_dimensions()

    # Calculate an ideal grid size based on the maze size and max cell size (in pixels)
    grid_width = maze_width // max_cell_size  # Number of columns in the grid
    grid_height = maze_height // max_cell_size  # Number of rows in the grid

    # Calculate step size for each grid cell
    x_step = maze_width / grid_width
    y_step = maze_height / grid_height

    font_scale = 0.3
    font_color = (255, 0, 0)
    font_thickness = 1

    # lines plus coordinates
    for i in range(grid_width + 1):  # horizontal grid
        for j in range(grid_height + 1):  # Vertical grid
            x = int(maze.topLeft.x + i * x_step)
            y = int(maze.topLeft.y + j * y_step)

            # Draw vertical grid lines
            if i < grid_width + 1:
                cv2.line(frame, (x, maze.topLeft.y), (x, maze.botLeft.y), (0, 255, 0), 1) # color and thickness

            # Draw horizontal grid lines
            if j < grid_height + 1 :
                cv2.line(frame, (maze.topLeft.x, y), (maze.topRight.x, y), (0, 255, 0), 1)

            # Display grid coordinates for each grid cell
            grid_coords = pixel_to_grid(x, y, maze, grid_size=(grid_width, grid_height))

            text_offset_x = 3  # Positive = move right
            text_offset_y = -3  # positive = move down

            cv2.putText(frame, f'{grid_coords[0]},{grid_coords[1]}', 
                        (x + text_offset_x, y + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                        font_color, font_thickness, cv2.LINE_AA)

# Corners stored
top_left_clicked = False
bottom_right_clicked = False

# Mouse clicks
def mouse_callback(event, x, y, flags, param):
    global top_left_clicked, bottom_right_clicked, maze

    if event == cv2.EVENT_LBUTTONDOWN:
        if not top_left_clicked:  # First click top left corner
            maze.topLeft = Point(x, y)
            top_left_clicked = True
            print(f"Top Left: {maze.topLeft}")
        elif not bottom_right_clicked:  # Second click bottom-right corner
            maze.botRight = Point(x, y)
            maze.topRight = Point(x, maze.topLeft.y)  # Same y as topLeft for topRight
            maze.botLeft = Point(maze.topLeft.x, y)  # Same x as topLeft for botLeft
            bottom_right_clicked = True
            print(f"Bottom Right: {maze.botRight}")

# Main function to process the camera feed
def main():
    global maze

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Camera resolution for maze
    camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize maze object
    maze = Maze()

    # Set mouse callback for user input
    cv2.namedWindow("Maze Setup with Grid")
    cv2.setMouseCallback("Maze Setup with Grid", mouse_callback)
    
    initial_size = 30
    cv2.createTrackbar("Cell Size", "Maze Setup with Grid", initial_size, 100, lambda x: None)  # Dummy function

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break

        size_val = cv2.getTrackbarPos("Cell Size", "Maze Setup with Grid")
        # Draw the grid and coordinates if corners are set
        draw_grid_and_coordinates(frame, maze, max_cell_size=size_val)  # Scale grid to fit maze

        # Show the frame with the maze and grid coordinates
        cv2.imshow("Maze Setup with Grid", frame)

        # Exit the loop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
