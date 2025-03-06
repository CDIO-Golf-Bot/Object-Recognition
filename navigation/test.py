import cv2
import numpy as np
from navigation.Maze import *
from navigation.Point import *

def main():
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

    maze_initialized = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        maze.draw_grid_and_coordinates(frame)

        # Only print grid points once the maze corners are set (initialized)
        if maze.topLeft and maze.botRight and not maze_initialized:
            maze_initialized = True
            maze.print_grid_points()  # Print all grid points

            # getting pixel point from a coordinate
            grid_x, grid_y = 5, 5 
            pixel_x, pixel_y = maze.get_point(grid_x, grid_y)
            print(f"Pixel coordinates for grid ({grid_x}, {grid_y}): ({pixel_x}, {pixel_y})")

        # Show the frame with the maze and grid coordinates
        cv2.imshow("Maze Setup with Grid", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()