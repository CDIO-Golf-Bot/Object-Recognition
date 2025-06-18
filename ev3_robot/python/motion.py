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
                   dist_thresh_cm=7.0,
                   early_stop_sec=0.5):
    """
    Drive straight toward (target_x_cm, target_y_cm) using only
    the robot_pose (vision) for distance & bearing.
    Stops 'early_stop_sec' seconds before the nominal threshold.
    """
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT

    # make sure we have a recent pose
    age = time.time() - robot_pose.get("timestamp", 0)
    if robot_pose["x"] is None or age > 1.0:
        print("No recent vision—cannot drive.")
        return

    # reset gyro to match vision heading
    hardware.gyro.reset()
    time.sleep(0.05)
    hardware.gyro_offset = robot_pose["theta"]

    # face the goal
    dx0 = target_x_cm - robot_pose["x"]
    dy0 = target_y_cm - robot_pose["y"]
    rotate_to_heading(utils.heading_from_deltas(dx0, dy0))

    # PD setup
    prev_error = 0.0
    LOOP_DT    = 0.01
    alpha      = 0.2
    smoothed   = 0.0

    # compute how much extra distance to stop early
    max_speed_cm_s = (speed_pct/100) * config.MAX_LINEAR_SPEED_CM_S
    extra_dist = early_stop_sec * max_speed_cm_s

    hardware.aux_motor.on(config.AUX_FORWARD_PCT)
    try:
        while True:
            # refresh vision
            age = time.time() - robot_pose["timestamp"]
            if robot_pose["x"] is None or age > 1.0:
                print("Lost vision—stopping.")
                break
            cx, cy = robot_pose["x"], robot_pose["y"]

            # pure-vision distance
            dist = math.hypot(target_x_cm - cx,
                              target_y_cm - cy)
            # inflate threshold by the "early-stop" margin
            tempDist = dist_thresh_cm + extra_dist
            if dist <= (dist_thresh_cm + extra_dist):
                print("Arrived early (dist {:.1f} <= {:.1f}).".format(dist, tempDist))
                break

            # compute heading correction
            desired = utils.heading_from_deltas(
                         target_x_cm - cx,
                         target_y_cm - cy)
            current_h = hardware.get_heading()
            error = ((desired - current_h + 180) % 360) - 180

            P = config.GYRO_KP * error
            D = config.GYRO_KD * (error - prev_error) / LOOP_DT
            prev_error = error
            raw_corr = max(-config.MAX_CORRECTION,
                           min(P + D, config.MAX_CORRECTION))
            smoothed = alpha*raw_corr + (1-alpha)*smoothed

            # drive
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

    # if we're GOTO + DELIVER in one, drive early-stop then deliver
    if 'goto' in cmd:
        x, y = cmd['goto']
        early = 1.0 if cmd.get('deliver') else 0.0
        drive_to_point(x, y, early_stop_sec=early)

    if cmd.get('deliver'):
        rotate_to_heading(0.0)
        _reverse_aux()

    if 'face' in cmd:
        rotate_to_heading(float(cmd['face']))