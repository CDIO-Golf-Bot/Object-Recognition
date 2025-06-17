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
    # Use vision pose when fresh, otherwise gyro
    if robot_pose["theta"] is not None and (time.time() - robot_pose["timestamp"]) < 0.5:
        return robot_pose["theta"]
    return hardware.get_heading()

# Drive straight

def drive_distance(distance_cm, speed_pct=None, target_angle=None, use_ultrasonic=False):
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT
    if target_angle is None:
        target_angle = get_corrected_heading()

    rotations = distance_cm / config.WHEEL_CIRC_CM
    last_error = 0.0

    start_dist = None
    if use_ultrasonic and ultrasonic_available:
        start_dist = distance_sensor.distance_centimeters
        print("Ultrasonic start: {:.2f} cm for target {:.2f} cm".format(start_dist, distance_cm))
    elif use_ultrasonic:
        print("Ultrasonic sensor data unavailable.")

    _start_aux()
    try:
        start_l = hardware.tank.left_motor.position
        start_r = hardware.tank.right_motor.position

        while True:
            # Encoder check
            pos_l = hardware.tank.left_motor.position - start_l
            pos_r = hardware.tank.right_motor.position - start_r
            avg_rot = (abs(pos_l) + abs(pos_r)) / 720.0  # 360 deg per rot
            if avg_rot >= rotations:
                break

            # Ultrasonic check
            if start_dist is not None:
                traveled = start_dist - distance_sensor.distance_centimeters
                if traveled >= distance_cm:
                    break

            # PID heading correction
            error = ((target_angle - hardware.get_heading() + 540) % 360) - 180
            derivative = error - last_error
            last_error = error
            corr = max(min(config.GYRO_KP * error + config.GYRO_KD * derivative,
                            config.MAX_CORRECTION), -config.MAX_CORRECTION)

            l_spd = max(min(speed_pct - corr + config.LEFT_BIAS, 100), -100)
            r_spd = max(min(speed_pct + corr, 100), -100)
            hardware.tank.on(SpeedPercent(l_spd), SpeedPercent(r_spd))
            time.sleep(0.01)
    finally:
        hardware.tank.off()
        _stop_aux()

    print("drive_distance done: rotations={:.2f} target={:.2f}".format(avg_rot, rotations))

# Rotate in place

def perform_turn(angle_deg):
    start_heading = get_corrected_heading()
    target = (start_heading + angle_deg) % 360.0
    _start_aux()
    try:
        while abs(((hardware.get_heading() - target + 540) % 360) - 180) > config.ANGLE_TOLERANCE:
            err = ((target - hardware.get_heading() + 540) % 360) - 180
            power = max(min(abs(err) * 0.4, config.TURN_SPEED_PCT), 5)
            if err > 0:
                hardware.tank.on(-power, power)
            else:
                hardware.tank.on(power, -power)
            time.sleep(0.01)
        hardware.tank.off()

        # Vision micro-correction
        if robot_pose["theta"] is not None and (time.time() - robot_pose["timestamp"]) < 0.5:
            resid = ((robot_pose["theta"] - hardware.get_heading() + 540) % 360) - 180
            if abs(resid) > config.ANGLE_TOLERANCE:
                print("Vision correct: {:.1f} deg".format(resid))
                adj = 5
                if resid > 0:
                    hardware.tank.on(-adj, adj)
                else:
                    hardware.tank.on(adj, -adj)
                time.sleep(abs(resid) / 30.0)
                hardware.tank.off()
    finally:
        hardware.tank.off()
        _stop_aux()

# Follow discrete path

def follow_path(points, start_heading_deg):
    hardware.gyro_offset = start_heading_deg
    hardware.calibrate_gyro()
    print("Gyro offset={}".format(hardware.gyro_offset))

    if robot_pose["x"] is not None and (time.time() - robot_pose["timestamp"]) < 0.5:
        cur_x = robot_pose["x"] / config.CELL_SIZE_CM
        cur_y = robot_pose["y"] / config.CELL_SIZE_CM
        cur_h = robot_pose["theta"]
    else:
        cur_x, cur_y = points[0]
        cur_h = get_corrected_heading()

    for nx, ny in points[1:]:
        if robot_pose["x"] is not None and (time.time() - robot_pose["timestamp"]) < 0.5:
            cur_x = robot_pose["x"] / config.CELL_SIZE_CM
            cur_y = robot_pose["y"] / config.CELL_SIZE_CM
            cur_h = robot_pose["theta"]

        dx = (nx - cur_x) * config.CELL_SIZE_CM
        dy = (ny - cur_y) * config.CELL_SIZE_CM
        tgt_h = math.degrees(math.atan2(dy, dx)) % 360

        # Drift correction
        live = get_corrected_heading()
        drift = ((live - cur_h + 540) % 360) - 180
        if abs(drift) > 5:
            print("Drift {} deg, correcting".format(drift))
            perform_turn(drift)
            cur_h = get_corrected_heading()

        # Turn to segment
        delta = ((tgt_h - cur_h + 540) % 360) - 180
        if abs(delta) > config.ANGLE_TOLERANCE:
            perform_turn(delta)
            cur_h = get_corrected_heading()

        dist = math.hypot(dx, dy)
        if dist > 0:
            drive_distance(dist, speed_pct=config.DRIVE_SPEED_PCT, target_angle=tgt_h)

        cur_x, cur_y, cur_h = nx, ny, tgt_h

# Pure Pursuit

def pure_pursuit_follow(path, lookahead_cm=15, speed_pct=None):
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT
    hardware.calibrate_gyro()

    def look_pt(pts, x, y, L):
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            dx, dy = x2 - x1, y2 - y1
            seg = math.hypot(dx, dy)
            if seg < 1e-6:
                continue
            t = ((x - x1) * dx + (y - y1) * dy) / (seg * seg)
            t = max(0, min(1, t))
            px, py = x1 + t * dx, y1 + t * dy
            if math.hypot(px - x, py - y) >= L:
                return px, py
        return pts[-1]

    _start_aux()
    try:
        while True:
            if robot_pose["x"] is not None and (time.time() - robot_pose["timestamp"]) < 0.5:
                x_cm, y_cm = robot_pose["x"], robot_pose["y"]
            else:
                x_cm, y_cm = path[0][0] * config.CELL_SIZE_CM, path[0][1] * config.CELL_SIZE_CM
            hdg = math.radians(hardware.get_heading())
            pt = look_pt([(px * config.CELL_SIZE_CM, py * config.CELL_SIZE_CM) for px, py in path], x_cm, y_cm, lookahead_cm)
            des = math.degrees(math.atan2(pt[1] - y_cm, pt[0] - x_cm))
            err = ((des - hardware.get_heading() + 540) % 360) - 180
            corr = max(min(config.GYRO_KP * err + config.GYRO_KD * err, config.MAX_CORRECTION), -config.MAX_CORRECTION)
            l = max(min(speed_pct - corr + config.LEFT_BIAS, 100), -100)
            r = max(min(speed_pct + corr, 100), -100)
            hardware.tank.on(SpeedPercent(l), SpeedPercent(r))
            time.sleep(0.01)
    finally:
        hardware.tank.off()
        _stop_aux()


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
                   dist_thresh_cm=2.0):
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT

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
    desired_heading = (360 - math.degrees(math.atan2(dy, dx))) % 360
    print("[drive_to_point] From ({:.2f}, {:.2f}) to ({:.2f}, {:.2f})".format(cur_x, cur_y, target_x_cm, target_y_cm))
    print("[drive_to_point] dx={:.2f}, dy={:.2f}, desired_heading={:.2f} deg".format(dx, dy, desired_heading))
    rotate_to_heading(desired_heading)

    _start_aux()
    try:
        last_x = cur_x
        last_y = cur_y

        while True:
            if (robot_pose["x"] is not None and
                (time.time() - robot_pose["timestamp"]) < 5.5):
                cur_x = robot_pose["x"]
                cur_y = robot_pose["y"]
                last_x, last_y = cur_x, cur_y
            else:
                if last_x is None:
                    print("No vision yet, stopping")
                    break
                print("No fresh vision, continuing with last known pose")
                cur_x, cur_y = last_x, last_y

            dx = target_x_cm - cur_x
            dy = target_y_cm - cur_y
            dist = math.hypot(dx, dy)
            if dist <= dist_thresh_cm:
                print("[drive_to_point] Arrived within {} cm of target".format(dist_thresh_cm))
                break

            desired = math.degrees(math.atan2(dy, dx)) % 360.0
            current = hardware.get_heading()
            error = ((desired - current + 540) % 360) - 180
            print("[drive_to_point] Current: {:.2f} deg, Desired: {:.2f} deg, Error: {:.2f} deg".format(current, desired, error))

            corr = max(min(config.GYRO_KP * error + config.GYRO_KD * error,
                           config.MAX_CORRECTION), -config.MAX_CORRECTION)

            l_spd = max(min(speed_pct - corr + config.LEFT_BIAS, 100), -100)
            r_spd = max(min(speed_pct + corr, 100), -100)

            print("[drive_to_point] l_spd={:.2f}, r_spd={:.2f}".format(l_spd, r_spd))
            hardware.tank.on(SpeedPercent(l_spd), SpeedPercent(r_spd))
            time.sleep(0.01)
    finally:
        hardware.tank.off()
        _stop_aux()


# Command handler
def handle_command(cmd, buf):
    if 'turn' in cmd:
        if buf.get('distance_buffer', 0) > 0:
            drive_distance(buf['distance_buffer'])
            buf['distance_buffer'] = 0
        perform_turn(float(cmd['turn']))

    if 'distance' in cmd:
        buf['distance_buffer'] = buf.get('distance_buffer', 0) + float(cmd['distance'])

    if cmd.get('deliver'):
        _reverse_aux()

    if 'goto' in cmd:
        if buf.get('distance_buffer', 0) > 0:
            drive_distance(buf['distance_buffer'])
            buf['distance_buffer'] = 0
        x, y = cmd['goto']
        drive_to_point(x, y)

    if 'face' in cmd:
        if buf.get('distance_buffer', 0) > 0:
            drive_distance(buf['distance_buffer'])
            buf['distance_buffer'] = 0
        rotate_to_heading(float(cmd['face']))