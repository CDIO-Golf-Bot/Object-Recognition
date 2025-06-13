#!/usr/bin/env python3
import time, math
import json
from ev3dev2.motor import SpeedPercent
import config as config, hardware


def _start_aux():    hardware.aux_motor.on(config.AUX_FORWARD_PCT)
def _stop_aux():     hardware.aux_motor.off()
def _reverse_aux():  
    hardware.aux_motor.on(config.AUX_REVERSE_PCT)
    time.sleep(config.AUX_REVERSE_SEC)
    hardware.aux_motor.off()


def drive_distance(distance_cm, speed_pct=None, target_angle=None):
    """PID-steer straight for given cm, using gyro for heading correction."""
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT
    if target_angle is None:
        target_angle = hardware.get_heading()

    rotations = distance_cm / config.WHEEL_CIRC_CM
    integral = last_error = 0.0

    _start_aux()
    try:
        start_l = hardware.tank.left_motor.position
        start_r = hardware.tank.right_motor.position

        while True:
            pos_l = hardware.tank.left_motor.position - start_l
            pos_r = hardware.tank.right_motor.position - start_r
            avg_rot = (abs(pos_l) + abs(pos_r)) / 2.0 / 360.0
            if avg_rot >= rotations:
                break

            # PID
            err = ((target_angle - hardware.get_heading() + 540) % 360) - 180
            if abs(err) < 1.0:
                err = 0.0
            integral = max(min(integral + err, 100), -100)
            derivative = err - last_error
            last_error = err

            corr = (
                config.GYRO_KP  * err +
                config.GYRO_KI  * integral +
                config.GYRO_KD  * derivative
            )
            corr = max(min(corr, config.MAX_CORRECTION), -config.MAX_CORRECTION)

            l_spd = max(min(speed_pct - corr + config.LEFT_BIAS, 100), -100)
            r_spd = max(min(speed_pct + corr, 100), -100)

            hardware.tank.on(SpeedPercent(l_spd), SpeedPercent(r_spd))
            time.sleep(0.01)
    finally:
        hardware.tank.off()
        _stop_aux()


def perform_turn(angle_deg):
    """Rotate in place by angle_deg (±), using gyro feedback."""
    target = (hardware.get_heading() + angle_deg) % 360.0
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
        time.sleep(0.1)
    finally:
        hardware.tank.off()
        _stop_aux()


def follow_path(points, start_heading_deg):
    """Given list of grid-cells and an initial heading, turn & drive each leg."""
    hardware.gyro_offset = start_heading_deg
    print("Gyro offset={:.1f} (gyro={})".format(hardware.gyro_offset, hardware.gyro.angle))
    cur_h = hardware.get_heading()
    cur_x, cur_y = points[0]

    for nx, ny in points[1:]:
        dx = (nx - cur_x) * config.CELL_SIZE_CM
        dy = (ny - cur_y) * config.CELL_SIZE_CM
        tgt_h = math.degrees(math.atan2(dy, dx)) % 360.0
        delta = ((tgt_h - cur_h + 180) % 360) - 180
        if abs(delta) > config.ANGLE_TOLERANCE:
            perform_turn(delta)
        dist = math.hypot(dx, dy)
        if dist > 0:
            drive_distance(dist, speed_pct=config.DRIVE_SPEED_PCT, target_angle=hardware.get_heading())
        cur_h, (cur_x, cur_y) = tgt_h, (nx, ny)


def handle_command(cmd, buf):
    """Process a single JSON command from the network."""
    if 'turn' in cmd:
        if buf['distance_buffer'] > 0:
            drive_distance(buf['distance_buffer'])
            buf['distance_buffer'] = 0
        perform_turn(float(cmd['turn']))

    if 'distance' in cmd:
        buf['distance_buffer'] += float(cmd['distance'])

    if cmd.get('deliver'):
        print("DELIVER!")
        _reverse_aux()
