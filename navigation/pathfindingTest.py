import cv2
import numpy as np
from navigation.Maze import *
from navigation.Point import *

def get_user_coordinates(grid_size):
    """Get valid grid coordinates from the user."""
    while True:
        try:
            x = int(input(f"Enter the x-coordinate (0 to {grid_size[0] - 1}): "))
            y = int(input(f"Enter the y-coordinate (0 to {grid_size[1] - 1}): "))
            if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                return (x, y)
            else:
                print(f"Coordinates must be between (0, 0) and ({grid_size[0] - 1}, {grid_size[1] - 1}). Try again.")
        except ValueError:
            print("Invalid input. Please enter integers.")

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
            print("Maze initialized.")

            # Get robot and ball positions from the user
            print("Enter the robot's position:")
            robot = get_user_coordinates(maze.gridSize)
            print("Enter the ball's position:")
            ball = get_user_coordinates(maze.gridSize)

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