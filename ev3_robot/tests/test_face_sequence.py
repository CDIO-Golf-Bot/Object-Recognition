#!/usr/bin/env python3
import time
from hardware import tank, get_heading, calibrate_gyro_aruco
from your_module import rotate_to_heading  # adjust to your module name

def test_face_sequence():
    # Optional: fuse gyro once at startup to whatever your last ArUco theta was
    # calibrate_gyro_aruco(robot_pose["theta"])

    test_headings = [0, 90, 180, 270, 45, 135, 225, 315]
    for target in test_headings:
        print(" Testing rotate_to_heading({})".format(target))
        rotate_to_heading(target)
        # give the motors a moment to settle
        time.sleep(0.2)
        actual = get_heading()
        test = ((actual - target + 180) % 360) - 180
        print("Ended at {:.1f} (error {:.1f})".format(actual, test))
        # pause so you can observe
        time.sleep(1.0)

if __name__ == "__main__":
    test_face_sequence()
