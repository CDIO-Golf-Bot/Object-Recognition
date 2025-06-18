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
    print("[rotate_to_heading] Target: {:.2f} deg (tolerance {} deg)".format(target_theta_deg, angle_thresh))
    try:
        while True:
            current = hardware.get_heading()

            error = utils.heading_error(target_theta_deg, current)
            print("[rotate_to_heading] Current: {:.2f} deg, Error: {:.2f} deg".format(current, error))

            if abs(error) <= angle_thresh:
                print("[rotate_to_heading] Heading within {} deg: {:.2f} deg".format(angle_thresh, current))
                break

            power = max(min(abs(error) * 0.5, config.TURN_SPEED_PCT), 5)
            if error > 0:
                print("[rotate_to_heading] Turning LEFT (CCW) with power {:.2f}".format(power))
                hardware.tank.on(power, -power)
            else:
                print("[rotate_to_heading] Turning RIGHT (CW) with power {:.2f}".format(power))
                hardware.tank.on(-power, power)
            time.sleep(0.01)
    finally:
        hardware.tank.off()




def drive_to_point(target_x_cm, target_y_cm, speed_pct=None, dist_thresh_cm=7.0):
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT

    # 1) ONE-TIME gyro fuse up front
    if robot_pose["theta"] is None:
        print("No ArUco heading—cannot drive.")
        return
    hardware.gyro.reset()
    time.sleep(0.05)
    hardware.gyro_offset = robot_pose["theta"]

    # 2) grab initial vision fix
    if robot_pose["x"] is None or (time.time() - robot_pose["timestamp"]) > 5.5:
        print("No vision yet—cannot drive.")
        return
    current_x, current_y = robot_pose["x"], robot_pose["y"]
    print("[drive_to_point] start at x={:.2f}, y={:.2f}".format(current_x, current_y))

    hardware.aux_motor.on(config.AUX_FORWARD_PCT)
    # 3) face the goal once
    dx = target_x_cm - current_x
    dy = target_y_cm - current_y
    initial_heading = utils.heading_from_deltas(dx, dy)
    print("[drive_to_point] initial dx={:.2f}, dy={:.2f}, heading={:.2f}".format(dx, dy, initial_heading))
    rotate_to_heading(initial_heading)

    # P-only controller setup
    prev_error = 0.0
    LOOP_DT    = 0.01
    STALE_LIMIT = 1.0  # seconds to wait for fresh vision
    alpha = 0.2          # smoothing factor, 0<α<1
    smoothed_corr = 0.0  # initialize


    def clamp(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    # keep track of last known good pose
    last_x, last_y = current_x, current_y
    last_x, last_y = current_x, current_y
    try:
        while True:
            # 0) wait for fresh vision
            age = time.time() - robot_pose["timestamp"]
            if robot_pose["x"] is None or age > STALE_LIMIT:
                hardware.tank.off()
                print("[drive_to_point] waiting for fresh vision (age={:.2f}s)".format(age))
                time.sleep(0.1)
                continue

            # 1) fresh vision: update pose
            current_x, current_y = robot_pose["x"], robot_pose["y"]
            last_x, last_y = current_x, current_y

            # 2) check arrival
            dx = target_x_cm - current_x
            dy = target_y_cm - current_y
            dist = math.hypot(dx, dy)
            print("[drive_to_point] dx={:.2f}, dy={:.2f}, dist={:.2f}".format(dx, dy, dist))
            if dist <= dist_thresh_cm:
                print("[drive_to_point] arrived within {} cm".format(dist_thresh_cm))
                break

            # 3) compute heading error and PD
            desired = utils.heading_from_deltas(dx, dy)
            current_heading = hardware.get_heading()
            error = ((desired - current_heading + 180) % 360) - 180
            print("[drive_to_point] desired={:.2f}, current={:.2f}, error={:.2f}".format(
                desired, current_heading, error
            ))

            P = config.GYRO_KP * error
            D = config.GYRO_KD * (error - prev_error) / LOOP_DT
            prev_error = error
            raw_corr = clamp(P + D, -config.MAX_CORRECTION, config.MAX_CORRECTION)
            smoothed_corr = alpha * raw_corr + (1 - alpha) * smoothed_corr
            corr = smoothed_corr
            print("[drive_to_point] P={:.2f}, D={:.2f}, corr={:.2f}".format(P, D, corr))

            # apply correction + bias + feed-forward
            raw_l = speed_pct + corr + config.LEFT_BIAS + config.FEED_FORWARD
            raw_r = speed_pct - corr + config.RIGHT_BIAS - config.FEED_FORWARD
            l_spd = clamp(raw_l, -100, 100)
            r_spd = clamp(raw_r, -100, 100)
            print("[drive_to_point] l_spd={:.2f}, r_spd={:.2f}".format(l_spd, r_spd))

            hardware.tank.on(SpeedPercent(l_spd), SpeedPercent(r_spd))
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