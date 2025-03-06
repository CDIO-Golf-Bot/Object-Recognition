import cv2
import numpy as np
from navigation.Maze import *
from navigation.Point import *

def main():
    """Main function to run the maze setup with camera feed and navigation."""
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

    # Flag to check if the maze is initialized
    maze_initialized = False

    # Define robot and ball positions (grid coordinates)
    robot = (2, 2)  # Example: Robot at (2, 2)
    ball = (8, 8)   # Example: Ball at (8, 8)

    # Variable to store the computed path
    path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        maze.draw_grid_and_coordinates(frame)

        # Only compute the path once the maze corners are set
        if maze.topLeft and maze.botRight and not maze_initialized:
            maze_initialized = True
            print("Maze initialized. Computing path...")

            # Find the shortest path from robot to ball
            path = maze.bfs(robot, ball)
            if path:
                print(f"Path found: {path}")
            else:
                print("No path found.")

        # Draw the path in every frame after the maze is initialized
        if maze_initialized and path:
            maze.draw_path(frame, path)

        # Show the frame with the maze and grid coordinates
        cv2.imshow("Maze Setup with Grid", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()