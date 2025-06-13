import threading
import cv2
from queue import Queue

from config import ROBOT_IP, ROBOT_PORT
from camera_io import capture_frames, display_frames
from detection import process_frames
from robot_comm import init_robot_connection, close_robot_connection
from navigation import ensure_outer_edges_walkable

# === Shared Globals ===
frame_queue = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
stop_event = threading.Event()

def main():
    print("🚀 Starting Object Recognition System")
    ensure_outer_edges_walkable()
    selected_goal = 'A'

    # 🔌 Robot connection
    init_robot_connection(ROBOT_IP, ROBOT_PORT)

    # 🎥 Threads for camera, detection, display
    cap_thread = threading.Thread(target=capture_frames, args=(frame_queue, stop_event))
    proc_thread = threading.Thread(target=process_frames, args=(frame_queue, output_queue, stop_event))
    disp_thread = threading.Thread(target=display_frames, args=(output_queue, stop_event))

    cap_thread.start()
    proc_thread.start()
    disp_thread.start()

    disp_thread.join()
    cap_thread.join()
    proc_thread.join()

    close_robot_connection()
    cv2.destroyAllWindows()
    print("✂️ Exiting cleanly")

if __name__ == "__main__":
    main()
