#!/usr/bin/env python3
import time
import math
import json
from ev3dev2.motor import SpeedPercent
import config
import hardware
import utils

# Attempt to initialize Ultrasonic Sensor (port from config)
try:
    from ev3dev2.sensor.lego import UltrasonicSensor
    from ev3dev2.sensor import INPUT_1
    distance_sensor = UltrasonicSensor(INPUT_1)
    ultrasonic_available = True
except Exception:
    print("Warning: Ultrasonic sensor not available.")
    distance_sensor = None
    ultrasonic_available = False

# Shared robot pose for fusion
robot_pose = {
    "x": None,
    "y": None,
    "theta": None,
    "timestamp": time.time()
}

# Internal helpers

def _start_aux():
    hardware.aux_motor.on(config.AUX_FORWARD_PCT)


def _stop_aux():
    hardware.aux_motor.off()


def _reverse_aux():
    hardware.aux_motor.on(config.AUX_REVERSE_PCT)
    time.sleep(config.AUX_REVERSE_SEC)
    hardware.aux_motor.off()

def calibrate_gyro_if_fresh():
    if robot_pose["theta"] is not None and (time.time() - robot_pose["timestamp"]) < 0.5:
        hardware.calibrate_gyro_aruco(robot_pose["theta"])
        print("Gyro calibrated to fresh ArUco pose.")
    else:
        print("Skipped gyro calibration: pose too old or missing.")

def rotate_to_heading(target_theta_deg, angle_thresh=1.5, overshoot_deg=0.0):
    """
    Rotate the robot in place until its heading matches target_theta_deg
    within angle_thresh degrees. Uses on_for_degrees for precision.
    """
    ROBOT_TRACK_CM = 12.5  # Distance between wheels (adjust for your robot)
    wheel_circ = config.WHEEL_CIRC_CM

    current = hardware.get_heading()
    error = utils.heading_error(target_theta_deg, current)
    if abs(error) <= angle_thresh:
        return

    # Apply overshoot in the turn direction
    direction = 1 if error > 0 else -1
    overshoot_target = (target_theta_deg + direction * overshoot_deg) % 360
    turn_angle = utils.heading_error(overshoot_target, current)

    # Calculate wheel degrees for the turn
    turn_circ = math.pi * ROBOT_TRACK_CM
    arc_len = (abs(turn_angle) / 360.0) * turn_circ
    wheel_degrees = (arc_len / wheel_circ) * 360

    left_speed = config.TURN_SPEED_PCT * direction
    right_speed = -config.TURN_SPEED_PCT * direction

    hardware.tank.on_for_degrees(
        SpeedPercent(left_speed), SpeedPercent(right_speed),
        abs(wheel_degrees), brake=True, block=True
    )

    # Optionally, check final heading and do a small correction if needed
    final_heading = hardware.get_heading()
    final_error = utils.heading_error(target_theta_deg, final_heading)
    if abs(final_error) > angle_thresh:
        # Recursively correct small error
        rotate_to_heading(target_theta_deg, angle_thresh, 0.0)


def drive_to_point(target_x_cm, target_y_cm, speed_pct=None, dist_thresh_cm=7.0):
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT

    if robot_pose["theta"] is None or robot_pose["x"] is None or robot_pose["y"] is None:
        print("No vision pose—cannot drive."); return

    calibrate_gyro_if_fresh()

    # Face the goal initially
    dx, dy = target_x_cm - robot_pose["x"], target_y_cm - robot_pose["y"]
    rotate_to_heading(utils.heading_from_deltas(dx, dy))

    while (robot_pose["x"] is None or (time.time() - robot_pose["timestamp"]) > config.MAX_ARUCO_AGE):
        print("[drive_to_point] Waiting for fresh vision pose after turn...")
        time.sleep(0.1)

    prev_error    = 0.0
    smoothed_corr = 0.0
    LOOP_DT       = 0.01
    alpha         = 0.4

    prev_l_spd = prev_r_spd = speed_pct
    SLEW_LIMIT = 5.0

    def clamp(v, lo, hi): return max(lo, min(v, hi))
    def slew(n, o, lim):
        d = n - o
        if   d >  lim: return o + lim
        if   d < -lim: return o - lim
        return n

    hardware.aux_motor.on(config.AUX_FORWARD_PCT)
    last_log = time.time()

    try:
        while True:
            # Wait for fresh vision pose
            while (robot_pose["x"] is None or (time.time() - robot_pose["timestamp"]) > config.MAX_ARUCO_AGE):
                print("[drive_to_point] Waiting for fresh vision pose...")
                time.sleep(0.2)

            now = time.time()
            if now - last_log >= config.LOG_INTERVAL:
                print("POSE: x={:.2f}, y={:.2f}, theta={:.1f}, ts={:.3f}, age={:.3f}".format(
                    robot_pose['x'],
                    robot_pose['y'],
                    robot_pose['theta'],
                    robot_pose['timestamp'],
                    now - robot_pose['timestamp']
                ))
                last_log = now

            # Always use vision pose for distance
            dist = math.hypot(target_x_cm - robot_pose["x"],
                              target_y_cm - robot_pose["y"])
            if dist <= dist_thresh_cm:
                print("Arrived (dist {:.1f} <= {:.1f})".format(dist, dist_thresh_cm))
                break

            # PD on heading (use gyro only)
            desired = utils.heading_from_deltas(
                        target_x_cm - robot_pose["x"],
                        target_y_cm - robot_pose["y"])
            
            curr_h = hardware.get_heading()
            heading_err = utils.heading_error(desired, curr_h)
            # Deadband: only re-align if error is significant
            if abs(heading_err) > config.ANGLE_OVERSHOOT:
                print("[drive_to_point] Heading error {:.1f} too large, re-aligning...".format(heading_err))
                rotate_to_heading(desired)
                time.sleep(0.2)
                continue  # restart loop with new heading

            # PD controller for heading correction
            error = ((desired - curr_h + 180) % 360) - 180
            P = config.GYRO_KP * error
            D = config.GYRO_KD * (error - prev_error) / LOOP_DT
            prev_error = error
            raw_corr = clamp(P + D, -config.MAX_CORRECTION, config.MAX_CORRECTION)
            smoothed_corr = alpha*raw_corr + (1-alpha)*smoothed_corr
            corr = smoothed_corr

            # Optionally slow down as you approach the target
            approach_factor = clamp(dist / config.APPROACH_DISTANCE, 0.3, 1.0)  # Slow down within 30cm
            base_speed = speed_pct * approach_factor

            # Compute & slew-limit speeds
            raw_l = base_speed + corr + config.LEFT_BIAS + config.FEED_FORWARD
            raw_r = base_speed - corr + config.RIGHT_BIAS - config.FEED_FORWARD
            l_spd = clamp(raw_l, -100, 100)
            r_spd = clamp(raw_r, -100, 100)

            l_spd = slew(l_spd, prev_l_spd, SLEW_LIMIT)
            r_spd = slew(r_spd, prev_r_spd, SLEW_LIMIT)
            prev_l_spd, prev_r_spd = l_spd, r_spd

            hardware.tank.on(SpeedPercent(l_spd),
                             SpeedPercent(r_spd))
            time.sleep(LOOP_DT)

    finally:
        hardware.tank.off()



# Command handler
def handle_command(cmd, buf):

    if 'distance' in cmd:
        buf['distance_buffer'] = buf.get('distance_buffer', 0) + float(cmd['distance'])

    if cmd.get('deliver'):
        rotate_to_heading(0.0)
        _reverse_aux()

    if 'goto' in cmd:
        x, y = cmd['goto']
        drive_to_point(x, y)

    if 'face' in cmd:
        rotate_to_heading(float(cmd['face']))