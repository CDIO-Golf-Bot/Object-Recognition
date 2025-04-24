import socket
import math
import json
import time

HOST = "10.225.58.57"
PORT = 12345
CM_PER_UNIT = 2.0
CARDINAL_ANGLE = {'E': 0.0, 'N': 90.0, 'W': 180.0, 'S': 270.0}

def shortest_angle_diff(target, current):
    diff = (target - current + 180) % 360 - 180
    return diff

def send_path(host, port, path, initial_heading):
    heading = CARDINAL_ANGLE[initial_heading]
    buffer_dist = 0.0  # accumulate straight‐line distance

    with socket.socket() as sock:
        sock.connect((host, port))
        current = path[0]

        for nxt in path[1:]:
            dx = (nxt[0] - current[0]) * CM_PER_UNIT
            dy = (nxt[1] - current[1]) * CM_PER_UNIT

            desired = math.degrees(math.atan2(dy, dx))
            δ = shortest_angle_diff(desired, heading)
            dist = math.hypot(dx, dy)

            if abs(δ) > 1e-6:
                # 1) If we were buffering a straight run, flush it now
                if buffer_dist > 0:
                    cmd = {"distance": buffer_dist}
                    sock.send((json.dumps(cmd) + "\n").encode())
                    print("Sent merged forward:", buffer_dist, "cm")
                    buffer_dist = 0.0

                # 2) Send the turn
                cmd = {"turn": δ}
                sock.send((json.dumps(cmd) + "\n").encode())
                print(f"Sent turn: {δ:.1f}°")
                heading = (heading + δ) % 360

            # If δ == 0, we’re still going straight—buffer it
            buffer_dist += dist
            current = nxt

        # Flush any final straight run
        if buffer_dist > 0:
            cmd = {"distance": buffer_dist}
            sock.send((json.dumps(cmd) + "\n").encode())
            print("Sent final merged forward:", buffer_dist, "cm")

        print("Done sending path.")

if __name__ == "__main__":
    # Example usage:
    path = [(1,1), (2,1), (3,1), (3,0), (3,-1), (2,-1)] #turn test
    #path = [(1,1), (2,1), (3,1), (4,1), (5,1), (6,1), (7,1)]     # forward test
    send_path(HOST, PORT, path, initial_heading='E')
