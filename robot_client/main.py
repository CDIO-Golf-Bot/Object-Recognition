# main.py
import threading
import cv2
from queue import Queue

from robot_client import config, robot_comm, navigation
from robot_client.camera_io import capture_frames, display_frames
from robot_client.detection import process_frames

def main():
    print("🚀 Starting Object Recognition System")

    # 1) Prepare navigation
    navigation.ensure_outer_edges_walkable()
    navigation.selected_goal = 'A'

    # 2) Shared stop flag for *all* threads
    stop_event = threading.Event()

    # 3) Fire up the connection-manager (tries every 5s)
    conn_thread = threading.Thread(
        target=robot_comm.connection_manager,
        args=(5.0, stop_event),
        name="ConnManagerThread",
        daemon=True
    )
    conn_thread.start()

    # 4) Queues between camera → inference → display
    capture_queue = Queue(maxsize=1)
    display_queue = Queue(maxsize=1)

    # 5) Worker threads
    capture_thread = threading.Thread(
        target=capture_frames,
        args=(capture_queue, stop_event),
        name="CaptureThread",
        daemon=True
    )
    inference_thread = threading.Thread(
        target=process_frames,
        args=(capture_queue, display_queue, stop_event),
        name="InferenceThread",
        daemon=True
    )
    display_thread = threading.Thread(
        target=display_frames,
        args=(display_queue, stop_event),
        name="DisplayThread",
        daemon=True
    )

    # 6) Start capture/inference/display
    capture_thread.start()
    inference_thread.start()
    display_thread.start()

    # 7) Wait for the UI to exit (user presses 'q')
    display_thread.join()

    # 8) Tell everything to shut down
    stop_event.set()
    capture_thread.join()
    inference_thread.join()

    # 9) Clean up network and windows
    robot_comm.close_robot_connection()
    cv2.destroyAllWindows()
    print("✂️ Exiting cleanly")

if __name__ == '__main__':
    main()
