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
        _stop_aux()


def drive_to_point(target_x_cm, target_y_cm, speed_pct=None,
                   dist_thresh_cm=4.0):
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT

    if robot_pose["theta"] is not None and (time.time() - robot_pose["timestamp"]) < 0.5:
        hardware.calibrate_gyro_aruco(robot_pose["theta"])

    # Get current position (wait for vision if needed)
    if (robot_pose["x"] is not None and
        (time.time() - robot_pose["timestamp"]) < 5.5):
        current_x = robot_pose["x"]
        current_y = robot_pose["y"]
    else:
        print("No vision yet, cannot drive")
        return
    
    _start_aux()

    dx = target_x_cm - current_x
    dy = target_y_cm - current_y
    initial_heading = utils.heading_from_deltas(dx, dy)
    print("[drive_to_point] From ({:.2f}, {:.2f}) to ({:.2f}, {:.2f})".format(current_x, current_y, target_x_cm, target_y_cm))
    print("[drive_to_point] dx={:.2f}, dy={:.2f}, desired_heading={:.2f} deg".format(dx, dy, initial_heading))
    rotate_to_heading(initial_heading)

    try:
        last_x = current_x
        last_y = current_y

        while True:
            if (robot_pose["x"] is not None and
                (time.time() - robot_pose["timestamp"]) < 5.5):
                current_x = robot_pose["x"]
                current_y = robot_pose["y"]
                last_x, last_y = current_x, current_y
            else:
                if last_x is None:
                    print("No vision yet, stopping")
                    break
                print("No fresh vision, continuing with last known pose")
                current_x, current_y = last_x, last_y

            dx = target_x_cm - current_x        # distance to x and y
            dy = target_y_cm - current_y
            dist    = math.hypot(dx, dy)       # pythagorean distance to target hypotenuse
            if dist <= dist_thresh_cm:
                print("[drive_to_point] Arrived within {} cm of target".format(dist_thresh_cm))
                break

            desired = utils.heading_from_deltas(dx, dy)
            current = hardware.get_heading()
            error   = utils.heading_error(desired, current)
            print("[drive_to_point] Current: {:.2f} deg, Desired: {:.2f} deg, Error: {:.2f} deg".format(current, desired, error))

            # Adjust correction strength based on distance
            steering_gain = min(max(dist / 50.0, 0.3), 1.0)  # Scale between 0.3–1.0
            corr = steering_gain * max(min(config.GYRO_KP * error + config.GYRO_KD * error,
                                           config.MAX_CORRECTION), -config.MAX_CORRECTION)

            l_spd = max(min(speed_pct - corr + config.LEFT_BIAS, 100), -100)
            r_spd = max(min(speed_pct + corr, 100), -100)

            # Optional: avoid dead zones
#            min_drive = 15
#            if abs(l_spd) < min_drive:
#                l_spd = min_drive * (1 if l_spd >= 0 else -1)
#            if abs(r_spd) < min_drive:
#                r_spd = min_drive * (1 if r_spd >= 0 else -1)

            print("[drive_to_point] l_spd={:.2f}, r_spd={:.2f}".format(l_spd, r_spd))
            hardware.tank.on(SpeedPercent(l_spd), SpeedPercent(r_spd))
            time.sleep(0.01)
    finally:
        hardware.tank.off()
        _stop_aux()



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