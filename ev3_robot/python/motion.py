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

# Heading fusion
def rotate_to_heading(target_theta_deg, angle_thresh=config.ANGLE_TOLERANCE):
    """
    Rotate the robot in place until its heading matches target_theta_deg
    within angle_thresh degrees.
    """
    gain = 1.0            # proportional gain
    min_power = 20        # minimum turn power (%)
    try:
        while True:
            current = hardware.get_heading()
            error = utils.heading_error(target_theta_deg, current)

            # stop when within tolerance
            if abs(error) <= angle_thresh:
                break

            # compute turn power
            power = max(min(abs(error) * gain, config.TURN_SPEED_PCT), min_power)

            if error > 0:
                hardware.tank.on(power, -power)
            else:
                hardware.tank.on(-power, power)

            # tight loop for snappier response
            time.sleep(0.005)
    finally:
        hardware.tank.off()


def drive_to_point(target_x_cm, target_y_cm,
                   speed_pct=None,
                   dist_thresh_cm=7.0):
    """
    Drive straight toward (target_x_cm, target_y_cm) using only
    the robot_pose (vision) for both distance and bearing.
    Heading correction still comes from the gyro.
    """
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT

    # 1) make sure we have a recent pose
    age = time.time() - robot_pose.get("timestamp", 0)
    if robot_pose["x"] is None or age > 1.0:
        print("No recent vision—cannot drive.")
        return

    # 2) reset gyro so heading matches ArUco theta
    hardware.gyro.reset()
    time.sleep(0.05)
    hardware.gyro_offset = robot_pose["theta"]

    # 3) face the goal once
    dx0 = target_x_cm - robot_pose["x"]
    dy0 = target_y_cm - robot_pose["y"]
    rotate_to_heading(utils.heading_from_deltas(dx0, dy0))

    # 4) PD controller setup
    prev_error = 0.0
    LOOP_DT    = 0.01
    alpha      = 0.2
    smoothed   = 0.0
    max_speed  = (speed_pct/100) * config.MAX_LINEAR_SPEED_CM_S

    hardware.aux_motor.on(config.AUX_FORWARD_PCT)
    try:
        while True:
            # refresh pose
            age = time.time() - robot_pose["timestamp"]
            if robot_pose["x"] is None or age > 1.0:
                print("Lost vision during drive—stopping.")
                break

            cx, cy = robot_pose["x"], robot_pose["y"]

            # 4a) distance check purely on vision pose
            dist = math.hypot(target_x_cm - cx,
                              target_y_cm - cy)
            if dist <= dist_thresh_cm:
                print("Arrived (dist {:.1f} <= {})".format(dist, dist_thresh_cm))
                break

            # 4b) compute desired bearing from pose
            desired = utils.heading_from_deltas(
                        target_x_cm - cx,
                        target_y_cm - cy)
            current_h = hardware.get_heading()
            error = ((desired - current_h + 180) % 360) - 180

            # PD terms
            P = config.GYRO_KP * error
            D = config.GYRO_KD * (error - prev_error) / LOOP_DT
            prev_error = error

            raw_corr = max(-config.MAX_CORRECTION,
                           min(P + D, config.MAX_CORRECTION))
            smoothed = alpha*raw_corr + (1-alpha)*smoothed

            # 4c) turn + forward
            left_spd  = speed_pct + smoothed + config.LEFT_BIAS
            right_spd = speed_pct - smoothed + config.RIGHT_BIAS

            hardware.tank.on(
                SpeedPercent(max(-100, min(100, left_spd))),
                SpeedPercent(max(-100, min(100, right_spd)))
            )

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