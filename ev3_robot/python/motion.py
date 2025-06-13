#!/usr/bin/env python3
import socket
import json
import math
import time

from ev3dev2.motor import SpeedPercent
import config
import hardware

def _start_aux():
    hardware.aux_motor.on(config.AUX_FORWARD_PCT)

def _stop_aux():
    hardware.aux_motor.off()

def _reverse_aux():
    hardware.aux_motor.on(config.AUX_REVERSE_PCT)
    time.sleep(config.AUX_REVERSE_SEC)
    hardware.aux_motor.off()

def drive_distance(distance_cm, speed_pct=None, target_angle=None):
    """
    PID-steer straight for given cm, using gyro for heading correction.
    """
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT
    if target_angle is None:
        target_angle = hardware.get_heading()

    rotations = distance_cm / config.WHEEL_CIRC_CM
    integral = 0.0
    last_error = 0.0

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

            corr = (config.GYRO_KP  * err +
                    config.GYRO_KI  * integral +
                    config.GYRO_KD  * derivative)
            corr = max(min(corr, config.MAX_CORRECTION), -config.MAX_CORRECTION)

            l_spd = max(min(speed_pct - corr + config.LEFT_BIAS, 100), -100)
            r_spd = max(min(speed_pct + corr, 100), -100)

            hardware.tank.on(SpeedPercent(l_spd), SpeedPercent(r_spd))
            time.sleep(0.01)
    finally:
        hardware.tank.off()
        _stop_aux()

def perform_turn(angle_deg):
    """
    Rotate in place by angle_deg (+/-), using gyro feedback.
    """
    target = (hardware.get_heading() + angle_deg) % 360.0
    _start_aux()
    try:
        while abs(((hardware.get_heading() - target + 540) % 360) - 180) > config.ANGLE_TOLERANCE:
            err = ((target - hardware.get_heading() + 540) % 360) - 180
            power = max(min(abs(err)*0.4, config.TURN_SPEED_PCT), 5)
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
    """
    Given list of grid-cells and an initial heading,
    turn & drive each leg.
    """
    hardware.gyro_offset = start_heading_deg
    print("Gyro offset={:.1f} (gyro={})".format(hardware.gyro_offset,
                                                hardware.gyro.angle))

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
            drive_distance(dist,
                           speed_pct=config.DRIVE_SPEED_PCT,
                           target_angle=hardware.get_heading())

        cur_h = tgt_h
        cur_x, cur_y = nx, ny

def handle_command(cmd, buf):
    """
    Process a single JSON command from the network.
    """
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

def run_server(host='', port=12345):
    """
    Start the TCP server to accept driving commands.
    """
    hardware.gyro.calibrate()
    hardware.gyro.reset()
    print("Gyro calibrated and zeroed.")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print("Listening on {}:{}...".format(host or '0.0.0.0', port))

        while True:
            conn, addr = srv.accept()
            print("Client connected from {}".format(addr))
            buf = b''
            buffer = {'distance_buffer': 0.0,
                      'last_cmd_time': time.time()}

            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        if buffer['distance_buffer'] > 0:
                            drive_distance(buffer['distance_buffer'])
                        break

                    buf += data
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        try:
                            cmd = json.loads(line.decode())
                            if 'path' in cmd and 'heading' in cmd:
                                run_heading = float(cmd['heading']) % 360.0
                                follow_path(cmd['path'], run_heading)
                            else:
                                handle_command(cmd, buffer)
                                buffer['last_cmd_time'] = time.time()
                        except json.JSONDecodeError:
                            print("Invalid JSON: {}".format(line))

                    # flush partial distance buffer if paused
                    if (buffer['distance_buffer'] > 0 and
                        time.time() - buffer['last_cmd_time'] > 0.2):
                        drive_distance(buffer['distance_buffer'])
                        buffer['distance_buffer'] = 0.0

            print("Client disconnected.")
