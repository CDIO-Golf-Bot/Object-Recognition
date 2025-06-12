#!/usr/bin/env python3
import socket
import json
import math
import time
from ev3dev2.motor import MoveTank, Motor, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedPercent
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_3

# === CONFIGURATION ===
WHEEL_DIAM_CM       = 3.7                    # your wheel diameter in cm
WHEEL_CIRC_CM       = WHEEL_DIAM_CM * math.pi
CELL_SIZE_CM        = 2.0                    # grid spacing (2cm)

# Gyro-driven PID parameters
gyro_kp = 2.5
gyro_ki = 0.0002
gyro_kd = 5.5
max_correction = 8

# Turn parameters
turn_speed_pct = 30    # maximum speed percent
angle_tolerance = 1.0  # degrees for stopping

# === INITIALIZE ===
tank = MoveTank(OUTPUT_B, OUTPUT_C)
aux_motor = Motor(OUTPUT_D)
gyro = GyroSensor(INPUT_3)

# Gyro calibration
def calibrate_gyro():
    print("Calibrating gyro, keep robot still...")
    gyro.mode = 'GYRO-CAL'
    time.sleep(1)
    gyro.mode = 'GYRO-ANG'
    time.sleep(0.1)

# Aux control
def _start_aux(): aux_motor.on(50)
def _stop_aux(): aux_motor.off()

# PID-corrected straight drive by distance
def drive_distance(distance_cm, speed_pct=30, target_angle=None):
    # Debug log: how many cells and cm
    cells = distance_cm / CELL_SIZE_CM
    # Use ASCII 'deg' instead of degree symbol
    print("Driving forward {:.0f} cells ({:.2f} cm) at heading {} deg".format(cells, distance_cm, gyro.angle))

    if target_angle is None:
        target_angle = gyro.angle

    degrees = (distance_cm / WHEEL_CIRC_CM) * 360.0
    tank.left_motor.position = 0
    tank.right_motor.position = 0

    integral = 0.0
    last_error = 0.0

    _start_aux()
    try:
        while True:
            left_pos  = abs(tank.left_motor.position)
            right_pos = abs(tank.right_motor.position)
            avg_pos = (left_pos + right_pos) / 2.0
            if avg_pos >= degrees:
                break

            # Gyro correction
            error = target_angle - gyro.angle
            if abs(error) < 1:
                error = 0
            integral += error
            derivative = error - last_error
            last_error = error

            corr = gyro_kp * error + gyro_ki * integral + gyro_kd * derivative
            corr = max(min(corr, max_correction), -max_correction)

            left_spd  = max(min(speed_pct - corr, 100), -100)
            right_spd = max(min(speed_pct + corr, 100), -100)

            tank.on(SpeedPercent(left_spd), SpeedPercent(right_spd))
            time.sleep(0.01)
    finally:
        tank.off()
        _stop_aux()

# Gyro-based turn
def perform_turn(angle_deg):
    # Debug log: turning direction
    direction = 'right' if angle_deg > 0 else 'left'
    # Use ASCII 'deg'
    print("Turning {} {:.2f} deg".format(direction, abs(angle_deg)))

    start_angle = gyro.angle
    target_angle = start_angle + angle_deg

    _start_aux()
    try:
        # proportional turn until within tolerance
        while abs(gyro.angle - target_angle) > angle_tolerance:
            error = target_angle - gyro.angle
            power = min(max(abs(error) * 0.3, 10), turn_speed_pct)
            if error > 0:
                tank.on(-power, power)
            else:
                tank.on(power, -power)
            time.sleep(0.01)

        tank.off()
        time.sleep(0.1)

        # final fine adjustment
        final_error = target_angle - gyro.angle
        if abs(final_error) > angle_tolerance:
            adj_power = 5
            if final_error > 0:
                tank.on(-adj_power, adj_power)
            else:
                tank.on(adj_power, -adj_power)
            time.sleep(abs(final_error) * 0.01)
    finally:
        tank.off()
        _stop_aux()

# Path following

def follow_path(points, start_heading_deg):
    cur_heading = start_heading_deg
    cur_x, cur_y = points[0]

    for next_x, next_y in points[1:]:
        dx = (next_x - cur_x) * CELL_SIZE_CM
        dy = (next_y - cur_y) * CELL_SIZE_CM

        # desired heading
        target_heading = math.degrees(math.atan2(dy, dx))
        if target_heading < 0:
            target_heading += 360.0

        # minimal turn
        delta = (target_heading - cur_heading + 180) % 360 - 180
        if abs(delta) > angle_tolerance:
            perform_turn(delta)

        # drive straight
        distance = math.hypot(dx, dy)
        if distance > 0.0:
            drive_distance(distance, speed_pct=30, target_angle=gyro.angle)

        cur_heading = target_heading
        cur_x, cur_y = next_x, next_y

# Command handling

def handle_command(cmd, buffer):
    if 'turn' in cmd:
        if buffer['distance_buffer'] > 0:
            drive_distance(buffer['distance_buffer'], speed_pct=30, target_angle=gyro.angle)
            buffer['distance_buffer'] = 0.0
        perform_turn(float(cmd['turn']))

    if 'distance' in cmd:
        buffer['distance_buffer'] += float(cmd['distance'])

# Server loop

def run_server(host='', port=12345):
    calibrate_gyro()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print("Listening on {}:{}...".format(host or '0.0.0.0', port))
        while True:
            conn, addr = srv.accept()
            print("Client connected from {}".format(addr))
            buf = b''
            buffer = {'distance_buffer': 0.0, 'last_cmd_time': time.time()}
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        if buffer['distance_buffer'] > 0:
                            drive_distance(buffer['distance_buffer'], speed_pct=30, target_angle=gyro.angle)
                        break
                    buf += data
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        try:
                            cmd = json.loads(line.decode())
                            if 'path' in cmd and 'heading' in cmd:
                                heading_map = {'E':0,'N':90,'W':180,'S':270}
                                start_deg = heading_map.get(cmd['heading'], 0)
                                follow_path(cmd['path'], start_deg)
                            else:
                                handle_command(cmd, buffer)
                                buffer['last_cmd_time'] = time.time()
                        except json.JSONDecodeError:
                            print("Invalid JSON: {}".format(line))

                    # auto-flush
                    if (buffer['distance_buffer'] > 0 and
                        time.time() - buffer['last_cmd_time'] > 0.2):
                        drive_distance(buffer['distance_buffer'], speed_pct=30, target_angle=gyro.angle)
                        buffer['distance_buffer'] = 0.0

if __name__ == '__main__':
    run_server()
