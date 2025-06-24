import threading
import cv2
import traceback
import math
import time
from queue import Empty

from robot_client import config, robot_comm, navigation, detection
from robot_client.navigation import planner, grid_utils, navigation_helpers


def capture_frames(frame_queue, stop_event):
    """Capture camera frames, tagging each with a timestamp and dropping old ones."""
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF,  "MSMF"),
        (cv2.CAP_ANY,   "ANY")
    ]
    cap = None
    for api, name in backends:
        try:
            tmp = cv2.VideoCapture(config.CAMERA_INDEX, api)
            if tmp.isOpened():
                print(f"Camera opened with {name} backend")
                cap = tmp
                break
            else:
                tmp.release()
        except Exception as e:
            print(f"Exception using {name} backend: {e}")

    if cap is None:
        print(f"Unable to open camera {config.CAMERA_INDEX}")
        stop_event.set()
        return

    # Apply camera settings
    cap.set(cv2.CAP_PROP_BUFFERSIZE, config.BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FPS,        config.FRAMES_PER_SEC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, config.CAMERA_BRIGHTNESS)

    # Log actual settings
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_f = cap.get(cv2.CAP_PROP_FPS)
    print(f"📷 Camera running at {actual_w:.0f}×{actual_h:.0f} @ {actual_f:.1f} FPS")

    cnt = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed; retrying...")
                continue

            cnt += 1
            if cnt % max(1, config.SKIP_FRAMES) != 0:
                continue

            timestamp = time.time()
            # drop any old frames
            try:
                while True:
                    frame_queue.get_nowait()
            except Empty:
                pass

            try:
                frame_queue.put((frame, timestamp), timeout=0.02)
            except Exception:
                pass

    except Exception:
        print("Unexpected error in capture_frames:")
        traceback.print_exc()
        stop_event.set()
    finally:
        cap.release()
        print("capture_frames exiting")


def display_frames(output_queue, stop_event):
    """Show frames with live FPS & latency, handle keypresses & clicks, plan & stream on 's'."""
    cv2.namedWindow("Live Object Detection")
    cv2.setMouseCallback("Live Object Detection", navigation.click_to_set_corners)

    fps = 0.0
    frame_counter = 0
    fps_timer = time.time()

    while not stop_event.is_set():
        try:
            frame, ts = output_queue.get(timeout=0.02)
            # flush intermediate frames, keep only latest
            try:
                while True:
                    frame, ts = output_queue.get_nowait()
            except Empty:
                pass
        except Empty:
            continue

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            planner.dynamic_route = True
            # Optionally, create a new stop_event for future runs:
            stop_event.clear()
            break
        elif key == ord('1'):
            navigation.selected_goal = 'A'
            print("Selected goal A")
        elif key == ord('2'):
            navigation.selected_goal = 'B'
            print("Selected goal B")
        elif key == ord('s'):
            planner.route_enabled = True
            planner.dynamic_route = False
            planner.plan_and_execute(stop_event)
            

        # Compute FPS & latency
        frame_counter += 1
        now = time.time()
        elapsed = now - fps_timer
        if elapsed >= 1.0:
            fps = frame_counter / elapsed
            frame_counter = 0
            fps_timer = now
        latency_ms = (now - ts) * 1000.0

        # Overlay text
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"LAT: {latency_ms:.0f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Live Object Detection", frame)

    cv2.destroyAllWindows()
    print("display_frames exiting")