#!/usr/bin/env python3
import socket
import json
import math
import time
from ev3dev2.motor import MoveTank, Motor, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedPercent
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_3

# === CONFIGURATION ===
WHEEL_DIAM_CM = 4.15
WHEEL_CIRC_CM = WHEEL_DIAM_CM * math.pi
CELL_SIZE_CM  = 2.0

gyro_kp = 2.5
gyro_ki = 0.0002
gyro_kd = 5.5
max_correction = 8

turn_speed_pct = 30
angle_tolerance = 1.0

# === INITIALIZE ===
tank = MoveTank(OUTPUT_B, OUTPUT_C)
aux_motor = Motor(OUTPUT_D)
gyro = GyroSensor(INPUT_3)

# Global offset to align gyro with client's heading
gyro_offset = 0.0

def calibrate_gyro():
    print("Calibrating gyro, keep robot still...")
    gyro.mode = 'GYRO-CAL'
    time.sleep(1)
    gyro.mode = 'GYRO-ANG'
    time.sleep(0.1)

def get_heading():
    return gyro.angle + gyro_offset

def _start_aux(): aux_motor.on(35)
def _stop_aux(): aux_motor.off()
def _reverse_aux(duration=1.5):
    aux_motor.on(-35)
    time.sleep(duration)
    aux_motor.off()

def drive_distance(distance_cm, speed_pct=30, target_angle=None):
    if target_angle is None:
        target_angle = get_heading()
    print("Driving {:.2f} cm at target {:.1f} deg (raw gyro: {:.1f})".format(
        distance_cm, target_angle, gyro.angle))

    rotations = distance_cm / WHEEL_CIRC_CM
    integral = 0.0
    last_error = 0.0

    _start_aux()
    try:
        tank.left_motor.position = 0
        tank.right_motor.position = 0

        while True:
            avg_rot = (abs(tank.left_motor.position) + abs(tank.right_motor.position)) / 2.0 / 360.0
            if avg_rot >= rotations:
                break

            error = target_angle - get_heading()
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

def perform_turn(angle_deg):
    direction = 'right' if angle_deg > 0 else 'left'
    print("Turning {} {:.2f} deg".format(direction, abs(angle_deg)))

    start_angle = get_heading()
    target_angle = start_angle + angle_deg

    _start_aux()
    try:
        while abs(get_heading() - target_angle) > angle_tolerance:
            error = target_angle - get_heading()
            power = min(max(abs(error) * 0.3, 10), turn_speed_pct)
            if error > 0:
                tank.on(-power, power)
            else:
                tank.on(power, -power)
            time.sleep(0.01)

        tank.off()
        time.sleep(0.1)

        final_error = target_angle - get_heading()
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


def follow_path(points, start_heading_deg):
    global gyro_offset
    gyro_offset = start_heading_deg - gyro.angle
    print("\nGyro offset applied: {:.1f} degrees (gyro: {:.1f}, client: {:.1f})\n".format(
        gyro_offset, gyro.angle, start_heading_deg))

    cur_heading = get_heading()  # <-- use adjusted gyro heading
    cur_x, cur_y = points[0]

    print("Path received from client:")
    for i, (x, y) in enumerate(points):
        print("  {:2d}: Grid ({}, {})".format(i, x, y))

    for next_x, next_y in points[1:]:
        dx = (next_x - cur_x) * CELL_SIZE_CM
        dy = (next_y - cur_y) * CELL_SIZE_CM

        target_heading = math.degrees(math.atan2(dy, dx)) % 360.0
        delta = (target_heading - cur_heading + 180) % 360 - 180
        if abs(delta) > angle_tolerance:
            perform_turn(delta)

        distance = math.hypot(dx, dy)
        if distance > 0.0:
            drive_distance(distance, speed_pct=30, target_angle=get_heading())

        cur_heading = target_heading
        cur_x, cur_y = next_x, next_y

def handle_command(cmd, buffer):
    if 'turn' in cmd:
        if buffer['distance_buffer'] > 0:
            drive_distance(buffer['distance_buffer'], speed_pct=30, target_angle=get_heading())
            buffer['distance_buffer'] = 0.0
        perform_turn(float(cmd['turn']))

    if 'distance' in cmd:
        buffer['distance_buffer'] += float(cmd['distance'])

    if 'deliver' in cmd and cmd['deliver'] == True:
        print("Executing delivery sequence.")
        _reverse_aux()

def run_server(host='', port=12345):
    calibrate_gyro()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print("Listening on {}:{}...".format(host or '0.0.0.0', port))
        while True:
            try:
                conn, addr = srv.accept()
                print("Client connected from {}".format(addr))
                buf = b''
                buffer = {'distance_buffer': 0.0, 'last_cmd_time': time.time()}
                with conn:
                    while True:
                        try:
                            data = conn.recv(1024)
                            if not data:
                                if buffer['distance_buffer'] > 0:
                                    drive_distance(buffer['distance_buffer'], speed_pct=30, target_angle=get_heading())
                                break
                            buf += data
                            while b'\n' in buf:
                                line, buf = buf.split(b'\n', 1)
                                try:
                                    cmd = json.loads(line.decode())
                                    if 'path' in cmd and 'heading' in cmd:
                                        start_deg = float(cmd['heading']) % 360.0
                                        follow_path(cmd['path'], start_deg)
                                    else:
                                        handle_command(cmd, buffer)
                                        buffer['last_cmd_time'] = time.time()
                                except json.JSONDecodeError:
                                    print("Invalid JSON: {}".format(line))
                            if (buffer['distance_buffer'] > 0 and
                                    time.time() - buffer['last_cmd_time'] > 0.2):
                                drive_distance(buffer['distance_buffer'], speed_pct=30, target_angle=get_heading())
                                buffer['distance_buffer'] = 0.0
                        except ConnectionResetError:
                            print("Client disconnected unexpectedly.")
                            break
                        except Exception as e:
                            print("Server error: {}".format(e))
                            break
            except Exception as e:
                print("Error accepting connection: {}".format(e))

if __name__ == '__main__':
    run_server()