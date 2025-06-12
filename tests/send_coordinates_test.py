#!/usr/bin/env python3
"""
Simple client script to send a hardcoded or computed path to the robot server.
Run: python client.py
"""
import socket
import json
import time

# === CONFIGURATION ===
HOST = '10.137.48.57'  # change to your robot's IP
PORT = 12345

# Hardcoded test path: square, 2cm grid coordinates (stop at last corner)
path = [(0, 0), (60, 0), (60, 20), (0, 20)]       # Y negative = left. Positive=right

# === Path filtering: remove intermediate colinear points ===
def compress_path(points):
    """
    Compress a list of grid points by removing intermediate points along straight segments.
    Keeps the endpoints of each linear run.
    """
    if len(points) < 3:
        return points[:]
    compressed = [points[0]]
    # compute direction from first to second
    prev_dx = points[1][0] - points[0][0]
    prev_dy = points[1][1] - points[0][1]
    # normalize direction to unit steps
    def norm(dx, dy):
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        return dx, dy
    prev_dir = norm(prev_dx, prev_dy)

    for curr, nxt in zip(points[1:], points[2:]):
        dx = nxt[0] - curr[0]
        dy = nxt[1] - curr[1]
        curr_dir = norm(dx, dy)
        # if direction changes, keep curr
        if curr_dir != prev_dir:
            compressed.append(curr)
        prev_dir = curr_dir
    # always include the last point
    compressed.append(points[-1])
    return compressed

# === Client send ===
def init_robot_connection(host, port, timeout=2.0):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        print(f"📡 Connected to robot at {host}:{port}")
        return sock
    except Exception as e:
        print(f"❌ Could not connect to robot: {e}")
        return None


def send_path(sock, grid_path, heading):
    """Send newline-delimited JSON so the EV3 server can parse it."""
    if not sock:
        print("⚠️  No robot connection, aborting send.")
        return
    # compress straight segments
    filtered = compress_path(grid_path)
    try:
        payload = {
            "heading": heading,
            "path": [[int(gx), int(gy)] for (gx, gy) in filtered]
        }
        data = json.dumps(payload).encode("utf-8") + b'\n'
        sock.sendall(data)
        print(f"📨 Sent path → {len(filtered)} points (compressed from {len(grid_path)}), heading={heading}")
    except Exception as e:
        print(f"❌ Failed to send path: {e}")

# === Main ===
if __name__ == "__main__":
    sock = init_robot_connection(HOST, PORT)
    heading = 'E'
    send_path(sock, path, heading)
    if sock:
        sock.close()
    print("✂️ Done sending path")
