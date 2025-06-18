#!/usr/bin/env python3
import time
import math
import json
from ev3dev2.motor import SpeedPercent
import config
import hardware
import utils
from HeadingFilter import HeadingFilter

# choose your blend: e.g. 90% gyro, 10% ArUco
heading_filter = HeadingFilter(alpha=0.9)

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




def drive_to_point(target_x_cm, target_y_cm, speed_pct=None, dist_thresh_cm=7.0):
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT

    # 1) one-time gyro reset
    if robot_pose["theta"] is None:
        print("No ArUco heading—cannot drive."); return
    hardware.gyro.reset()
    time.sleep(0.05)
    hardware.gyro_offset = robot_pose["theta"]

    # 2) initial vision fix
    if robot_pose["x"] is None or (time.time() - robot_pose["timestamp"]) > 1.0:
        print("No recent vision—cannot drive."); return
    current_x, current_y = robot_pose["x"], robot_pose["y"]

    # 3) face the goal
    dx, dy = target_x_cm - current_x, target_y_cm - current_y
    rotate_to_heading(utils.heading_from_deltas(dx, dy))

    # controller & motion setup
    prev_error    = 0.0
    smoothed_corr = 0.0
    LOOP_DT       = 0.01
    alpha         = 0.2
    max_speed_cm_s= (config.DRIVE_SPEED_PCT/100)*config.MAX_LINEAR_SPEED_CM_S

    # odometry state
    last_tick_l = hardware.tank.left_motor.position
    last_tick_r = hardware.tank.right_motor.position

    prev_l_spd = prev_r_spd = speed_pct
    SLEW_LIMIT = 5.0

    def clamp(v, lo, hi): return max(lo, min(v, hi))
    def slew(n, o, lim):
        d = n - o
        if   d >  lim: return o + lim
        if   d < -lim: return o - lim
        return n

    hardware.aux_motor.on(config.AUX_FORWARD_PCT)

    try:
        while True:
            # 0) dead-reckon from encoders
            new_l = hardware.tank.left_motor.position
            new_r = hardware.tank.right_motor.position
            d_l = new_l - last_tick_l; d_r = new_r - last_tick_r
            last_tick_l, last_tick_r = new_l, new_r
            dcm_l = d_l * config.WHEEL_CIRC_CM / config.TICKS_PER_REV
            dcm_r = d_r * config.WHEEL_CIRC_CM / config.TICKS_PER_REV
            dcenter = (dcm_l + dcm_r)/2
            θ = math.radians(hardware.get_heading())
            current_x += dcenter * math.cos(θ)
            current_y += dcenter * math.sin(θ)

            # 1) vision fuse if fresh
            age = time.time() - robot_pose["timestamp"]
            if age < 0.5:
                # simple override: take vision pose
                current_x, current_y = robot_pose["x"], robot_pose["y"]

            # 2) stopping check (no waiting!)
            dist = math.hypot(target_x_cm - current_x,
                              target_y_cm - current_y)
            # inflate for lag:
            effective_thresh = dist_thresh_cm + max_speed_cm_s * age
            if dist <= effective_thresh:
                print("Arrived (dist {:.1f} <= {:.1f})".format(dist, effective_thresh))
                break

            # 3) PD on heading
            desired = utils.heading_from_deltas(
                        target_x_cm - current_x,
                        target_y_cm - current_y)
            curr_h = hardware.get_heading()
            error = ((desired - curr_h + 180) % 360) - 180
            P = config.GYRO_KP * error
            D = config.GYRO_KD * (error - prev_error) / LOOP_DT
            prev_error = error
            raw_corr = clamp(P + D, -config.MAX_CORRECTION, config.MAX_CORRECTION)
            smoothed_corr = alpha*raw_corr + (1-alpha)*smoothed_corr
            corr = smoothed_corr

            # 4) compute & slew-limit speeds
            raw_l = speed_pct + corr + config.LEFT_BIAS + config.FEED_FORWARD
            raw_r = speed_pct - corr + config.RIGHT_BIAS - config.FEED_FORWARD
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