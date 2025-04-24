#!/usr/bin/env python3
import socket
import json
import math
import time
from ev3dev2.motor import MoveTank, OUTPUT_B, OUTPUT_C

# === CONFIGURATION ===
WHEEL_DIAM_CM = 3.6                    # your wheel diameter
WHEEL_CIRC_CM = WHEEL_DIAM_CM * 3.1416
TRACK_CM      = 24                    # distance between the two wheels
TURN_SPEED    = 30                     # deg/s for in-place turns
DRIVE_SPEED   = 50                     # deg/s for driving straight

# === INITIALIZE ===
tank = MoveTank(OUTPUT_B, OUTPUT_C)

# Add a new function to drive straight for a given total distance
def drive_distance(d):
    wheel_deg = (d / WHEEL_CIRC_CM) * 360.0
    tank.on_for_degrees(DRIVE_SPEED, DRIVE_SPEED, wheel_deg)

def handle_command(cmd):
    """
    cmd is a dict possibly containing:
      - "turn":   signed degrees (robot frame, +CW or +CCW)
      - "distance": distance in cm to drive straight
    """
    # 1) In-place turn
    if "turn" in cmd:
        θ = float(cmd["turn"])  # desired robot turn angle in degrees
        # Convert robot rotation angle into wheel degrees
        wheel_deg = (TRACK_CM * math.pi * abs(θ)) / WHEEL_CIRC_CM
        if θ > 0:
            tank.on_for_degrees(TURN_SPEED, -TURN_SPEED, wheel_deg)
        else:
            tank.on_for_degrees(-TURN_SPEED, TURN_SPEED, wheel_deg)

    # 2) Straight drive
    if "distance" in cmd:
        d = float(cmd["distance"])
        wheel_deg = (d / WHEEL_CIRC_CM) * 360.0
        tank.on_for_degrees(DRIVE_SPEED, DRIVE_SPEED, wheel_deg)

def run_server(host='', port=12345):
    buf = b''
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
            with conn:
                distance_buffer = 0.0
                last_cmd_time = time.time()

                while True:
                    data = conn.recv(1024)
                    if not data:
                        print("Client disconnected.")
                        if distance_buffer > 0:
                            print("Flushing remaining distance: {distance_buffer:.1f} cm")
                            drive_distance(distance_buffer)
                        break

                    buf += data
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        try:
                            cmd = json.loads(line.decode())
                            print("Command received:", cmd)

                            if "turn" in cmd:
                                # Flush distance first before turning
                                if distance_buffer > 0:
                                    print("Flushing buffered distance: {distance_buffer:.1f} cm")
                                    drive_distance(distance_buffer)
                                    distance_buffer = 0.0

                                θ = float(cmd["turn"])
                                wheel_deg = (TRACK_CM * math.pi * abs(θ)) / WHEEL_CIRC_CM
                                if θ > 0:
                                    tank.on_for_degrees(TURN_SPEED, -TURN_SPEED, wheel_deg)
                                else:
                                    tank.on_for_degrees(-TURN_SPEED, TURN_SPEED, wheel_deg)

                            if "distance" in cmd:
                                d = float(cmd["distance"])
                                distance_buffer += d
                                last_cmd_time = time.time()

                        except json.JSONDecodeError:
                            print("Invalid JSON:", line)

                    # Check if it's been a short pause (e.g. 0.3s) with no new command
                    if distance_buffer > 0 and time.time() - last_cmd_time > 0.3:
                        print("Auto-flushing after pause: {distance_buffer:.1f} cm")
                        drive_distance(distance_buffer)
                        distance_buffer = 0.0


if __name__ == '__main__':
    run_server()
