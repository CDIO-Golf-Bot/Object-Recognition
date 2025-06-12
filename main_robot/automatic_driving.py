#!/usr/bin/env python3
import socket
import json
import math
import time
from ev3dev2.motor import MoveTank, Motor, OUTPUT_B, OUTPUT_C, OUTPUT_D

# === CONFIGURATION ===
WHEEL_DIAM_CM       = 3.7                    # your wheel diameter in cm
WHEEL_CIRC_CM       = WHEEL_DIAM_CM * math.pi
TRACK_CM            = 24                     # effective distance between wheels in cm
TURN_SPEED          = 35                     # deg/s for in-place turns
DRIVE_SPEED         = 50                     # deg/s for driving straight
TURN_CALIBRATION    = 1.1                    # multiply to tune actual vs. commanded turn
AUX_MOTOR_SPEED     = 50                     # deg/s for D motor whenever moving
CELL_SIZE_CM = 2.0

def follow_path(points, start_heading_deg):
    """
    points: list of (x,y) tuples
    start_heading_deg: robot's initial heading in degrees (0=east, 90=north)
    """
    cur_heading = start_heading_deg
    (cur_x, cur_y) = points[0]

    for (next_x, next_y) in points[1:]:
        dx = next_x - cur_x
        dy = next_y - cur_y

        # atan2 returns radians with 0=+x axis
        target_heading = math.degrees(math.atan2(dy, dx))
        # normalize into [0,360)
        if target_heading < 0:
            target_heading += 360

        # compute minimal turn difference in [-180,180]
        delta = (target_heading - cur_heading + 180) % 360 - 180

        # do the turn and drive
        if abs(delta) > 1e-6:
            perform_turn(delta)

        distance_cm = math.hypot(dx, dy) * CELL_SIZE_CM
        if distance_cm > 1e-6:
            drive_distance(distance_cm)

        # update for next segment
        cur_heading = target_heading
        cur_x, cur_y = next_x, next_y

# === INITIALIZE ===
tank = MoveTank(OUTPUT_B, OUTPUT_C)
aux_motor = Motor(OUTPUT_D)

def _start_aux():
    """Start the D-motor spinning forward at AUX_MOTOR_SPEED."""
    aux_motor.on(AUX_MOTOR_SPEED)

def _stop_aux():
    """Stop the D-motor."""
    aux_motor.off()

# Drive straight for a given total distance (cm)
def drive_distance(distance_cm):
    wheel_deg = (distance_cm / WHEEL_CIRC_CM) * 360.0
    # start aux before moving
    _start_aux()
    tank.on_for_degrees(DRIVE_SPEED, DRIVE_SPEED, wheel_deg)
    # stop aux when done
    _stop_aux()

# Perform an in-place turn for a desired robot angle (deg)
def perform_turn(theta_deg):
    base_wheel_deg = (TRACK_CM * math.pi * abs(theta_deg)) / WHEEL_CIRC_CM
    calibrated_deg = base_wheel_deg * TURN_CALIBRATION
    # start aux before moving
    _start_aux()
    if theta_deg > 0:
        tank.on_for_degrees(TURN_SPEED, -TURN_SPEED, calibrated_deg)
    else:
        tank.on_for_degrees(-TURN_SPEED, TURN_SPEED, calibrated_deg)
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
                            if 'path' in cmd and 'heading' in cmd:
                                heading_map = {'E': 0, 'N': 90, 'W': 180, 'S': 270}
                                start_deg = heading_map.get(cmd['heading'], 0)
                                follow_path(cmd['path'], start_deg)
                            else:
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
