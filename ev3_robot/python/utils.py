import math


def normalize_angle(angle):
    """
    Normalize any angle (in degrees) into [0, 360).
    """
    return angle % 360


def heading_from_deltas(dx, dy):
    """
    Compute a compass-style heading (0–360°) from dx, dy offsets.

    - Uses atan2(dy, dx) to get the angle from +X axis (math coords).
    - Converts to degrees and then shifts to compass:
        0° = north (positive Y), increasing clockwise.
    """
    raw_deg = math.degrees(math.atan2(dy, dx))
    # Convert math-angle (0°=+X, CCW-positive) to compass (0°=+Y, CW-positive)
    # First normalize raw to [0,360), then compass adjust
    return (360 - (raw_deg % 360) + 90) % 360


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
