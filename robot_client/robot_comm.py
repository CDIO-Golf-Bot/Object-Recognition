# robot_comm.py
#
# Open/close socket to the robot and send it a compressed path payload.


import socket
import json

from robot_client.config import ROBOT_HEADING
from robot_client.navigation.planner import compress_path

robot_sock = None


def init_robot_connection(ip: str, port: int, timeout=2.0):
    global robot_sock
    try:
        robot_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        robot_sock.settimeout(timeout)
        robot_sock.connect((ip, port))
        robot_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        print(f"📡 Connected to robot at {ip}:{port}")
    except Exception as e:
        print(f"❌ Could not connect to robot: {e}")
        robot_sock = None

def close_robot_connection():
    global robot_sock
    if robot_sock:
        robot_sock.close()
        print("🔌 Robot connection closed.")


def send_pose(x_cm, y_cm, theta_deg):
    global robot_sock
    if robot_sock is None:
        print("⚠️ No robot connection, cannot send pose.")
        return
    try:
        pose_msg = {
            "pose": {
                "x": float(round(x_cm, 2)),
                "y": float(round(y_cm, 2)),
                "theta": float(round(theta_deg, 1))
            }
        }
        data = json.dumps(pose_msg).encode("utf-8") + b'\n'
        robot_sock.sendall(data)
        print("📡 Sent pose: x={:.2f}, y={:.2f}, θ={:.1f}".format(x_cm, y_cm, theta_deg))
    except Exception as e:
        print("❌ Failed to send pose: {}".format(e))


def send_path(grid_path: list, heading: float = ROBOT_HEADING):
    global robot_sock
    if robot_sock is None:
        print("⚠️ No robot connection, aborting send.")
        return
    try:
        filtered = compress_path(grid_path)
        payload = {
            "heading": heading,
            "path": [[int(gx), int(gy)] for (gx, gy) in filtered]
        }
        data = json.dumps(payload).encode("utf-8") + b'\n'
        robot_sock.sendall(data)
        print(f"📤 Sent path → {len(filtered)} points (compressed from {len(grid_path)}), heading={heading:.1f}")

        # Deliver command
        deliver_cmd = json.dumps({"deliver": True}).encode("utf-8") + b'\n'
        robot_sock.sendall(deliver_cmd)
        print("📨 Sent deliver command")
    except Exception as e:
        print(f"❌ Failed to send path: {e}")
