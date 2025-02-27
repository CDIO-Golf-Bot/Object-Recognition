import cv2
import numpy as np

from points import Point  # Importing the Point class from points.py

class Maze:
    def __init__(self, topLeft, topRight, botLeft, botRight):
        self.topLeft = topLeft
        self.topRight = topRight
        self.botLeft = botLeft
        self.botRight = botRight

    def get_corners(self):
        return self.topLeft, self.topRight, self.botLeft, self.botRight

def mark_maze(frame, maze: Maze):
    topLeft, topRight, botLeft, botRight = maze.get_corners()  
    points = [(topLeft.x, topLeft.y), (topRight.x, topRight.y),
              (botRight.x, botRight.y), (botLeft.x, botLeft.y)]

    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]  # Connect the last point to the first
        cv2.line(frame, start_point, end_point, (0, 0, 255), 2)  # color, thicknes

    # Draw the corner points
    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)  # Green circle at the points

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # hardcoded points
    topLeft = Point(100, 100)
    topRight = Point(500, 100)
    botLeft = Point(100, 400)
    botRight = Point(500, 400)

    maze = Maze(topLeft, topRight, botLeft, botRight) 

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        mark_maze(frame, maze)

        cv2.imshow("Maze Setup", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
