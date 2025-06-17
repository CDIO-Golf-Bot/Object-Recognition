import socket
import json

from robot_client.config import ROBOT_HEADING
from robot_client.navigation.planner import compress_path

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
def send_pose(x_cm, y_cm, theta_deg) -> bool:
    """
    Send a one-off pose update to the robot via send_cmd().
    Returns True on success, False on failure.
    """
    pose_msg = {
        "pose": {
            "x": float(round(x_cm, 2)),
            "y": float(round(y_cm, 2)),
            "theta": float(round(theta_deg, 1))
        }
    }
    json_str = json.dumps(pose_msg)
    print("📥 Pose → {}".format(json_str))

    ok = send_cmd(pose_msg)   # this closes & nulls socket if it fails
    if not ok:
        print("⚠️ Pose send failed; will reconnect on next send_cmd.")
    return ok



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

def send_face(theta_deg: float):
    """
    Tell the robot to rotate until vision/gyro says we're facing the target heading.
    Retries once on connection failure.
    """
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
            print("❌ face retry also failed.")