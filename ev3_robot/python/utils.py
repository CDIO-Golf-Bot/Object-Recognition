import math


def normalize_angle(angle):
    """
    Normalize any angle (in degrees) into [0, 360).
    """
    return angle % 360

import math

def heading_from_deltas(dx, dy):
    """
    Compute heading (0–360°) in the robot/gyro frame, when:
      • world +X is right (east), 
      • world +Y is down  (south),
      • robot 0° is +X (east), increasing CCW.
    """
    # Flip Y so that world-down → math-up
    raw = math.degrees(math.atan2(-dy, dx))
    # Normalize into [0, 360)
    return raw % 360



def heading_error(target, current):
    """
    Compute the minimal signed error (in degrees) to rotate from current to target.
    Result is in the interval (-180, 180].
    """
    return ((target - current + 540) % 360) - 180


def compute_offset(aruco_heading, raw_heading):
    """
    Compute the gyro offset so that raw_heading + offset == aruco_heading (mod 360).
    """
    return ((aruco_heading - normalize_angle(raw_heading)) % 360)
