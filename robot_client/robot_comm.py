import socket
import json
import time

from robot_client.config import ROBOT_HEADING
from robot_client.navigation.planner import compress_path
from robot_client.config import ROBOT_IP, ROBOT_PORT
import config

robot_sock = None


def init_robot_connection(ip, port, timeout=2.0):
    global robot_sock
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        s.settimeout(None)                     # << unblock mode
        s.setsockopt(socket.SOL_SOCKET,
                     socket.SO_KEEPALIVE, 1)
        robot_sock = s
        print(f"📡 Connected to robot at {ip}:{port}")
    except Exception as e:
        print(f"❌ Could not connect to robot: {e}")
        robot_sock = None


def close_robot_connection():
    global robot_sock
    if robot_sock:
        robot_sock.close()
        print("🔌 Robot connection closed.")

def send_pose(x_cm, y_cm, theta_deg, timestamp = None) -> bool:
    """
    Send a one-off pose update to the robot via send_cmd().
    Returns True on success, False on failure.
    """
    if robot_sock is None:
        return False
    pose = {
        "x":     float(round(x_cm, 2)),
        "y":     float(round(y_cm, 2)),
        "theta": float(round(theta_deg, 1))
    }
    # include timestamp if provided
    if timestamp is not None:
        pose["timestamp"] = float(timestamp)
    pose_msg = {"pose": pose}
    json_str = json.dumps(pose_msg)
    print("📥 Pose → {}".format(json_str))

    ok = send_cmd(pose_msg)   # this closes & nulls socket if it fails
    if not ok:
        print("⚠️ Pose send failed; will reconnect on next send_cmd.")
    return ok

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


# NEW DIRECT DRIVING COMMANDS

def send_cmd(cmd: dict) -> bool:
    """
    Serialize a single JSON command and send it over the socket.
    Returns True on success, False on failure.
    """
    global robot_sock
    if not robot_sock:
        print("⚠️ No robot connection, cannot send command.")
        return False

    data = (json.dumps(cmd) + "\n").encode("utf-8")
    try:
        robot_sock.sendall(data)
        return True
    except Exception as e:
        print(f"❌ Failed to send {cmd!r}: {e!s}")
        # force reconnect next time
        try:
            robot_sock.close()
        except Exception:
            pass
        robot_sock = None
        return False

def send_goto(x_cm: float, y_cm: float):
    """
    Tell the robot to drive continuously until vision/gyro says we've reached (x,y).
    Retries once on connection failure.
    """
    cmd = {"goto": [float(x_cm), float(y_cm)]}
    ok = send_cmd(cmd)
    if ok:
        print(f"📡 Sent goto → x={x_cm:.1f}cm, y={y_cm:.1f}cm")
    else:
        print("⚠️ goto failed—attempting to reconnect and retry")
        init_robot_connection(config.ROBOT_IP, config.ROBOT_PORT)
        if send_cmd(cmd):
            print(f"📡 Sent goto (retry) → x={x_cm:.1f}cm, y={y_cm:.1f}cm")
        else:
            print("❌ goto retry also failed.")
""" 
def send_face(theta_deg: float):
    cmd = {"face": float(theta_deg)}
    ok = send_cmd(cmd)
    if ok:
        print(f"📡 Sent face → θ={theta_deg:.1f}°")
    else:
        print("⚠️ face failed—attempting to reconnect and retry")
        init_robot_connection(config.ROBOT_IP, config.ROBOT_PORT)
        if send_cmd(cmd):
            print(f"📡 Sent face (retry) → θ={theta_deg:.1f}°")
        else:
            print("❌ face retry also failed.") """


def connection_manager(interval: float, stop_event):
    """
    Runs in its own thread.  Every `interval` seconds,
    if we're not connected, try to init_robot_connection().
    """
    while not stop_event.is_set():
        if robot_sock is None:
            init_robot_connection(ROBOT_IP, ROBOT_PORT)
        time.sleep(interval)