#!/usr/bin/env python3
import socket
import json
import math
import time
from ev3dev2.motor import MoveTank, OUTPUT_B, OUTPUT_C

# === CONFIGURATION ===
WHEEL_DIAM_CM       = 3.7                    # your wheel diameter in cm
WHEEL_CIRC_CM       = WHEEL_DIAM_CM * math.pi
TRACK_CM            = 24                     # effective distance between wheels in cm
TURN_SPEED          = 35                     # deg/s for in-place turns
DRIVE_SPEED         = 50                     # deg/s for driving straight
TURN_CALIBRATION    = 1.1                    # multiply to tune actual vs. commanded turn

# === INITIALIZE ===
tank = MoveTank(OUTPUT_B, OUTPUT_C)

# Drive straight for a given total distance (cm)
def drive_distance(distance_cm):
    wheel_deg = (distance_cm / WHEEL_CIRC_CM) * 360.0
    tank.on_for_degrees(DRIVE_SPEED, DRIVE_SPEED, wheel_deg)

# Perform an in-place turn for a desired robot angle (deg)
def perform_turn(theta_deg):
    base_wheel_deg = (TRACK_CM * math.pi * abs(theta_deg)) / WHEEL_CIRC_CM
    calibrated_deg = base_wheel_deg * TURN_CALIBRATION
    if theta_deg > 0:
        tank.on_for_degrees(TURN_SPEED, -TURN_SPEED, calibrated_deg)
    else:
        tank.on_for_degrees(-TURN_SPEED, TURN_SPEED, calibrated_deg)

# Handle a single parsed command dict
# cmd keys: 'turn' (deg), 'distance' (cm)
def handle_command(cmd, buffer):
    # buffer: dict with 'distance_buffer'
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
# Accept one client at a time; keep listening after disconnects
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
                    if buffer['distance_buffer'] > 0 and time.time() - buffer['last_cmd_time'] > 0.2:
                        drive_distance(buffer['distance_buffer'])
                        buffer['distance_buffer'] = 0.0

if __name__ == '__main__':
    run_server()
