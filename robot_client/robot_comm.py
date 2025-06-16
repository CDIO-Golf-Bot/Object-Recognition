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

def send_path(grid_path: list, heading: float = ROBOT_HEADING):
    filtered = compress_path(grid_path)
    payload = {
        "heading": heading,
        "path": [[int(gx), int(gy)] for (gx, gy) in filtered]
    }
    data = json.dumps(payload).encode("utf-8") + b'\n'
    robot_sock.sendall(data)
    print("Sent full path → {} points, heading={:.1f}"
          .format(len(filtered), heading))
    # this also sends the deliver command under the hood:
    deliver_cmd = json.dumps({"deliver": True}).encode("utf-8") + b'\n'
    robot_sock.sendall(deliver_cmd)


def send_pose(x_cm, y_cm, theta_deg):
    """Send a one-off pose update to the robot."""
    global robot_sock
    if robot_sock is None:
        print("⚠️ No robot connection, cannot send pose.")
        return
    pose_msg = {
        "pose": {
            "x": float(round(x_cm, 2)),
            "y": float(round(y_cm, 2)),
            "theta": float(round(theta_deg, 1))
        }
    }
    data = json.dumps(pose_msg).encode('utf-8') + b'\n'
    try:
        robot_sock.sendall(data)
        #print(f"📡 Sent pose: x={pose_msg['pose']['x']:.2f}, y={pose_msg['pose']['y']:.2f}, θ={pose_msg['pose']['theta']:.1f}")
    except Exception as e:
        print(f"❌ Failed to send pose: {e}")


def send_turn(angle_deg: float):
    """Send a turn command to the robot."""
    global robot_sock
    if robot_sock is None:
        print("⚠️ No robot connection, cannot send turn.")
        return
    cmd = {"turn": float(angle_deg)}
    data = json.dumps(cmd).encode('utf-8') + b'\n'
    try:
        robot_sock.sendall(data)
        print(f"📡 Sent turn: {angle_deg:.1f}°")
    except Exception as e:
        print(f"❌ Failed to send turn: {e}")


def send_distance(distance_cm: float):
    """Send a drive-distance command to the robot."""
    global robot_sock
    if robot_sock is None:
        print("⚠️ No robot connection, cannot send distance.")
        return
    cmd = {"distance": float(distance_cm)}
    data = json.dumps(cmd).encode('utf-8') + b'\n'
    try:
        robot_sock.sendall(data)
        print(f"📡 Sent distance: {distance_cm:.1f}cm")
    except Exception as e:
        print(f"❌ Failed to send distance: {e}")


def send_deliver():
    """Send the deliver command after routing is complete."""
    global robot_sock
    if robot_sock is None:
        print("⚠️ No robot connection, cannot send deliver.")
        return
    cmd = {"deliver": True}
    data = json.dumps(cmd).encode('utf-8') + b'\n'
    try:
        robot_sock.sendall(data)
        print("📡 Sent deliver command")
    except Exception as e:
        print(f"❌ Failed to send deliver: {e}")
