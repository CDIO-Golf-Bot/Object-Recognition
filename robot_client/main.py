# main.py
# Entry point: wires capture, inference (ArUco+YOLO), and display threads
# Setup project on pc: pip install -e .

import threading
import cv2
from queue import Queue

from robot_client import config, robot_comm, navigation
from robot_client.camera_io import capture_frames, display_frames
from robot_client.detection import process_frames


def main():
    print("🚀 Starting Object Recognition System")

    # Ensure navigation grid is set up
    navigation.ensure_outer_edges_walkable()
    navigation.selected_goal = 'A'

    # Initialize robot connection
    robot_comm.init_robot_connection(config.ROBOT_IP, config.ROBOT_PORT)

    # Shared stop flag
    stop_event = threading.Event()

    # Queues between stages
    capture_queue = Queue(maxsize=1)
    display_queue = Queue(maxsize=1)

    # Thread: Capture frames & ArUco pose
    capture_thread = threading.Thread(
        target=capture_frames,
        args=(capture_queue, stop_event),
        name="CaptureThread",
        daemon=True
    )

    # Thread: YOLO inference + annotation
    inference_thread = threading.Thread(
        target=process_frames,
        args=(capture_queue, display_queue, stop_event),    
        name="InferenceThread",
        daemon=True
    )

    # Thread: Display & user interaction
    display_thread = threading.Thread(
        target=display_frames,
        args=(display_queue, stop_event),
        name="DisplayThread",
        daemon=True
    )

    # Start threads
    capture_thread.start()
    inference_thread.start()
    display_thread.start()

    # Wait for display thread to finish (user presses 'q')
    display_thread.join()

    # Signal all threads to stop
    stop_event.set()

    # Cleanup remaining threads
    capture_thread.join()
    inference_thread.join()

    # Close robot connection and windows
    robot_comm.close_robot_connection()
    cv2.destroyAllWindows()
    print("✂️ Exiting cleanly")


if __name__ == '__main__':
    main()
