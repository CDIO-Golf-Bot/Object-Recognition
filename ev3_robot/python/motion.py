#!/usr/bin/env python3
import time
import math
import json
from ev3dev2.motor import SpeedPercent
import config
import hardware

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

def get_corrected_heading():
    return hardware.get_heading()

def rotate_to_heading(target_theta_deg, angle_thresh=10.0):
    print("[rotate_to_heading] Target: {:.2f} deg (tolerance {} deg)".format(target_theta_deg, angle_thresh))
    _start_aux()
    try:
        while True:
            if (robot_pose["theta"] is not None and
                (time.time() - robot_pose["timestamp"]) < 5.5):
                current = robot_pose["theta"]
            else:
                current = hardware.get_heading()

            error = ((target_theta_deg - current + 540) % 360) - 180
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

    if robot_pose["theta"] is not None and (time.time() - robot_pose["timestamp"]) < 1.0:
        hardware.gyro_offset = robot_pose["theta"]
        print("[gyro] Synced: gyro_offset = {:.2f}".format(hardware.gyro_offset))

    # Get current position (wait for vision if needed)
    if (robot_pose["x"] is not None and
        (time.time() - robot_pose["timestamp"]) < 5.5):
        cur_x = robot_pose["x"]
        cur_y = robot_pose["y"]
    else:
        print("No vision yet, cannot drive")
        return

    dx = target_x_cm - cur_x
    dy = target_y_cm - cur_y
    initial_heading = (360 - math.degrees(math.atan2(dy, dx))) % 360
    print("[drive_to_point] From ({:.2f}, {:.2f}) to ({:.2f}, {:.2f})".format(cur_x, cur_y, target_x_cm, target_y_cm))
    print("[drive_to_point] dx={:.2f}, dy={:.2f}, desired_heading={:.2f} deg".format(dx, dy, initial_heading))
    rotate_to_heading(initial_heading)

    _start_aux()
    try:
        last_x, last_y = cur_x, cur_y
        while True:
            # … update cur_x, cur_y …

            dx = target_x_cm - cur_x
            dy = target_y_cm - cur_y
            dist = math.hypot(dx, dy)
            if dist <= dist_thresh_cm:
                print("[drive_to_point] Arrived within {} cm".format(dist_thresh_cm))
                break

            # live heading to target, _same_ formula as before
            desired = math.degrees(math.atan2(dy, dx)) % 360
            current = hardware.get_heading()
            error = ((desired - current + 540) % 360) - 180

            steering_gain = min(max(dist / 50.0, 0.3), 1.0)
            raw = config.GYRO_KP * error + config.GYRO_KD * error
            corr = steering_gain * max(min(raw, config.MAX_CORRECTION),
                                      -config.MAX_CORRECTION)
            # if corr is steering the wrong way, invert here:
            # corr = -corr

            r_speed = speed_pct - corr + config.LEFT_BIAS
            l_speed = speed_pct + corr

            # enforce dead-zone
            min_drive = 15
            if abs(r_speed) < min_drive:
                r_speed = min_drive * (1 if r_speed >= 0 else -1)
            if abs(l_speed) < min_drive:
                l_speed = min_drive * (1 if l_speed >= 0 else -1)

            print("[drive_to_point] r_speed={:.1f}, l_speed={:.1f}".format(r_speed,l_speed ))
            hardware.tank.on(SpeedPercent(l_speed), SpeedPercent(r_speed))
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
        # Use vision pose when fresh, otherwise gyro
        x, y = cmd['goto']
        drive_to_point(x, y)

    if 'face' in cmd:
        rotate_to_heading(float(cmd['face']))