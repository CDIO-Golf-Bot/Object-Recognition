#!/usr/bin/env python3
import socket
import json
import math
import time
from ev3dev2.motor import MoveTank, Motor, OUTPUT_B, OUTPUT_C, OUTPUT_D
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_1

# === CONFIGURATION ===
WHEEL_DIAM_CM       = 3.7                    # your wheel diameter in cm
WHEEL_CIRC_CM       = WHEEL_DIAM_CM * math.pi
TRACK_CM            = 24                     # effective distance between wheels in cm
TURN_SPEED          = 35                     # deg/s for in-place turns
DRIVE_SPEED         = 50                     # deg/s for driving straight
TURN_CALIBRATION    = 1.1                    # multiply to tune actual vs. commanded turn
AUX_MOTOR_SPEED     = 50                     # deg/s for D motor whenever moving
GYRO_CORRECTION_GAIN = 1.2                   # PID gain for gyro correction during straight driving
GYRO_SAMPLE_TIME    = 0.01                   # seconds between gyro readings

# === INITIALIZE ===
tank = MoveTank(OUTPUT_B, OUTPUT_C)
aux_motor = Motor(OUTPUT_D)
gyro = GyroSensor(INPUT_1)

def _start_aux():
    """Start the D-motor spinning forward at AUX_MOTOR_SPEED."""
    aux_motor.on(AUX_MOTOR_SPEED)

def _stop_aux():
    """Stop the D-motor."""
    aux_motor.off()

def _gyro_reset():
    """Reset the gyro sensor and wait for calibration."""
    gyro.mode = 'GYRO-RATE'
    gyro.mode = 'GYRO-ANG'
    time.sleep(0.1)

def _gyro_get_angle():
    """Get the current gyro angle in degrees."""
    return gyro.angle

def _gyro_get_rate():
    """Get the current rotation rate in degrees/second."""
    return gyro.rate

# Drive straight for a given total distance (cm) using gyro correction
def drive_distance(distance_cm):
    wheel_deg = (distance_cm / WHEEL_CIRC_CM) * 360.0
    _gyro_reset()
    start_angle = _gyro_get_angle()
    
    # Calculate total time needed for the move
    move_time = abs(wheel_deg) / DRIVE_SPEED
    start_time = time.time()
    
    # start aux before moving
    _start_aux()
    
    while time.time() - start_time < move_time:
        current_angle = _gyro_get_angle()
        angle_error = current_angle - start_angle
        correction = angle_error * GYRO_CORRECTION_GAIN
        
        # Apply correction to motor speeds
        left_speed = DRIVE_SPEED - correction
        right_speed = DRIVE_SPEED + correction
        
        # Constrain speeds to reasonable values
        left_speed = max(min(left_speed, DRIVE_SPEED * 1.5), DRIVE_SPEED * 0.5)
        right_speed = max(min(right_speed, DRIVE_SPEED * 1.5), DRIVE_SPEED * 0.5)
        
        tank.on(left_speed, right_speed)
        time.sleep(GYRO_SAMPLE_TIME)
    
    tank.off()
    # stop aux when done
    _stop_aux()

# Perform an in-place turn for a desired robot angle (deg) using gyro
def perform_turn(theta_deg):
    _gyro_reset()
    target_angle = _gyro_get_angle() + theta_deg
    angle_tolerance = 2.0  # degrees
    
    # start aux before moving
    _start_aux()
    
    while abs(_gyro_get_angle() - target_angle) > angle_tolerance:
        current_error = target_angle - _gyro_get_angle()
        
        # Simple proportional control
        turn_power = min(max(abs(current_error) * 0.5, 10), TURN_SPEED)
        
        if current_error > 0:
            tank.on(turn_power, -turn_power)
        else:
            tank.on(-turn_power, turn_power)
        
        time.sleep(GYRO_SAMPLE_TIME)
    
    tank.off()
    # stop aux when done
    _stop_aux()

# Handle a single parsed command dict
# cmd keys: 'turn' (deg), 'distance' (cm)
def handle_command(cmd, buffer):
    # 1) If there's a turn, flush any buffered forward first
    if 'turn' in cmd:
        if buffer['distance_buffer'] > 0:
            drive_distance(buffer['distance_buffer'])
            buffer['distance_buffer'] = 0.0
        perform_turn(float(cmd['turn']))

    # 2) Buffer distance commands
    if 'distance' in cmd:
        buffer['distance_buffer'] += float(cmd['distance'])

# Main server loop
def run_server(host='', port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print("Listening on {}:{}...".format(host or '0.0.0.0', port))
        while True:
            print("Waiting for client...")
            conn, addr = srv.accept()
            print("Client connected from", addr)
            buf = b''
            buffer = {'distance_buffer': 0.0, 'last_cmd_time': time.time()}
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        print("Client disconnected.")
                        # flush any remaining forward
                        if buffer['distance_buffer'] > 0:
                            drive_distance(buffer['distance_buffer'])
                        break
                    buf += data
                    # process newline-delimited JSON commands
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        try:
                            cmd = json.loads(line.decode())
                            print("Command received:", cmd)
                            handle_command(cmd, buffer)
                            buffer['last_cmd_time'] = time.time()
                        except json.JSONDecodeError:
                            print("Invalid JSON:", line)
                    # auto-flush forward if idle > threshold
                    if (buffer['distance_buffer'] > 0 and
                        time.time() - buffer['last_cmd_time'] > 0.2):
                        drive_distance(buffer['distance_buffer'])
                        buffer['distance_buffer'] = 0.0

if __name__ == '__main__':
    run_server()