#!/usr/bin/env python3
import socket
import json
import math
import time
from ev3dev2.motor import MoveTank, Motor, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedPercent
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_4

# === CONFIGURATION ===
WHEEL_DIAM_CM = 4.13
WHEEL_CIRC_CM = WHEEL_DIAM_CM * math.pi
CELL_SIZE_CM  = 2.0

gyro_kp = 3.0
gyro_ki = 0.0003
gyro_kd = 6.0
max_correction = 12

turn_speed_pct = 30
angle_tolerance = 1.0

LEFT_BIAS = 5.0

# === INITIALIZE ===
tank = MoveTank(OUTPUT_B, OUTPUT_C)
aux_motor = Motor(OUTPUT_D)
gyro = GyroSensor(INPUT_4)

# Global offset to align gyro with client's heading
gyro_offset = 0.0

def get_heading():
    return (gyro.angle + gyro_offset) % 360.0

def calibrate_gyro():
    print("Calibrating gyro, keep robot still...")
    gyro.mode = 'GYRO-CAL'
    time.sleep(1)
    gyro.mode = 'GYRO-ANG'
    time.sleep(0.1)
    gyro.reset()
    time.sleep(0.1)


def _start_aux(): aux_motor.on(35)
def _stop_aux(): aux_motor.off()

def _reverse_aux(duration=1.5):
    aux_motor.on(-35)
    time.sleep(duration)
    aux_motor.off()

def drive_distance(distance_cm, speed_pct=30, target_angle=None):
    """
    Drive straight for a given distance in cm using gyro-based correction (PID).
    Uses motor rotation feedback to determine when to stop.
    """
    if target_angle is None:
        target_angle = get_heading()
    target_angle = target_angle % 360.0  # Normalize

    print("Driving {:.2f} cm at target {:.1f} (current heading: {:.1f})".format(
        distance_cm, target_angle, get_heading()))

    rotations = distance_cm / WHEEL_CIRC_CM
    integral = 0.0
    last_error = 0.0

    _start_aux()
    try:
        start_left = tank.left_motor.position
        start_right = tank.right_motor.position

        while True:
            current_left = tank.left_motor.position
            current_right = tank.right_motor.position
            avg_rot = (abs(current_left - start_left) + abs(current_right - start_right)) / 2.0 / 360.0
            if avg_rot >= rotations:
                break

            # Gyro correction logic
            error = (target_angle - get_heading() + 540) % 360 - 180
            if abs(error) < 1.0:
                error = 0.0

            integral += error
            derivative = error - last_error
            last_error = error

            # Clamp integral to avoid windup
            integral = max(min(integral, 100), -100)

            correction = gyro_kp * error + gyro_ki * integral + gyro_kd * derivative
            correction = max(min(correction, max_correction), -max_correction)

            left_speed = max(min(speed_pct - correction + LEFT_BIAS, 100), -100)
            right_speed = max(min(speed_pct + correction, 100), -100)

            tank.on(SpeedPercent(left_speed), SpeedPercent(right_speed))
            time.sleep(0.01)
    finally:
        tank.off()
        _stop_aux()

        # Debug: Actual traveled distance and drift
        actual_left_rot = (tank.left_motor.position - start_left) / 360.0
        actual_right_rot = (tank.right_motor.position - start_right) / 360.0
        actual_avg_rot = (abs(actual_left_rot) + abs(actual_right_rot)) / 2.0
        actual_distance = actual_avg_rot * WHEEL_CIRC_CM
        drift = (actual_right_rot - actual_left_rot) * WHEEL_CIRC_CM

        print("Actual distance: {:.2f} cm (L: {:.2f}, R: {:.2f})".format(
            actual_distance,
            actual_left_rot * WHEEL_CIRC_CM,
            actual_right_rot * WHEEL_CIRC_CM
        ))
        print("Drift (Right - Left): {:.2f} cm\n".format(drift))



def perform_turn(angle_deg):
    direction = 'right' if angle_deg > 0 else 'left'
    print("Turning {} {:.2f} deg".format(direction, abs(angle_deg)))

    start_angle = get_heading()
    target_angle = (start_angle + angle_deg) % 360.0

    _start_aux()
    try:
        while abs((get_heading() - target_angle + 540) % 360 - 180) > angle_tolerance:
            error = (target_angle - get_heading() + 540) % 360 - 180
            k_p = 0.4
            min_power = 5
            power = max(min(abs(error) * k_p, turn_speed_pct), min_power)

            if error > 0:
                tank.on(-power, power)
            else:
                tank.on(power, -power)
            time.sleep(0.01)

        tank.off()
        time.sleep(0.1)

        final_error = (target_angle - get_heading() + 540) % 360 - 180
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
    gyro_offset = start_heading_deg  # The gyro starts at 0 after calibration
    print("\nGyro offset applied: {:.1f} degrees (gyro: {:.1f}, client: {:.1f})\n".format(
        gyro_offset, gyro.angle, start_heading_deg))

    print("Path received from client:")
    for i, (x, y) in enumerate(points):
        print("  {:2d}: Grid ({}, {})".format(i, x, y))

    cur_heading = get_heading()
    cur_x, cur_y = points[0]

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
