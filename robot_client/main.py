# main.py
#
# Kick off threads for camera capture, object detection, and display;
# manage robot connection and clean shutdown.


import threading
import cv2
from queue import Queue

import config
import camera_io
import detection
import robot_comm
import navigation

# === Shared Globals ===
frame_queue  = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
stop_event   = threading.Event()

def main():
    print("🚀 Starting Object Recognition System")
    navigation.ensure_outer_edges_walkable()
    # selected_goal lives in navigation.selected_goal if you need to set it:
    navigation.selected_goal = 'A'

    # 🔌 Robot connection
    robot_comm.init_robot_connection(config.ROBOT_IP, config.ROBOT_PORT)

    # 🎥 Threads for camera, detection, display
    cap_thread = threading.Thread(
        target=camera_io.capture_frames,
        args=(frame_queue, stop_event)
    )
    proc_thread = threading.Thread(
        target=detection.process_frames,
        args=(frame_queue, output_queue, stop_event)
    )
    disp_thread = threading.Thread(
        target=camera_io.display_frames,
        args=(output_queue, stop_event)
    )

    cap_thread.start()
    proc_thread.start()
    disp_thread.start()

    disp_thread.join()
    cap_thread.join()
    proc_thread.join()

    robot_comm.close_robot_connection()
    cv2.destroyAllWindows()
    print("✂️ Exiting cleanly")

if __name__ == "__main__":
    main()
