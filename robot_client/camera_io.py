import threading
import cv2
import traceback
import math
import time

from robot_client import config, robot_comm, navigation, detection
from robot_client.navigation.planner import compute_best_route


def capture_frames(frame_queue, stop_event):
    """Capture camera frames, trying multiple Windows backends."""
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
                print("Camera opened with {} backend".format(name))
                cap = tmp
                break
            else:
                tmp.release()
                print("Could not open camera with {} backend".format(name))
        except Exception as e:
            print("Exception using {} backend: {}".format(name, e))

    if cap is None:
        print("Unable to open camera {}".format(config.CAMERA_INDEX))
        stop_event.set()
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, config.BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FPS,        config.FRAMES_PER_SEC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

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
            if cnt % config.SKIP_FRAMES == 0:
                try:
                    frame_queue.put(frame, timeout=0.02)
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
    """Show frames, handle keypresses & mouse clicks, plan & stream on 's'."""
    cv2.namedWindow("Live Object Detection")
    cv2.setMouseCallback("Live Object Detection", navigation.click_to_set_corners)

    # ─── FPS tracking ──────────────────────────────────────────────────────────
    fps = 0.0
    frame_counter = 0
    fps_timer = time.time()

    while not stop_event.is_set():
        frame = None
        try:
            frame = output_queue.get(timeout=0.02)
        except Exception:
            pass

        # Handle keypress
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break
        elif key == ord('1'):
            navigation.selected_goal = 'A'
            print("Selected goal A")
        elif key == ord('2'):
            navigation.selected_goal = 'B'
            print("Selected goal B")
        elif key == ord('s'):
            balls = detection.ball_positions_cm
            if not balls:
                print("No balls detected; cannot plan route.")
            else:
                # 1️⃣ Plan the full route (in cm & grid cells)
                route_cm, grid_cells = compute_best_route(
                    balls,
                    navigation.selected_goal
                )

                if route_cm:
                    navigation.pending_route = route_cm
                    navigation.full_grid_path = grid_cells
                    print("Planned route: {} waypoints, {} grid points"
                          .format(len(route_cm), len(grid_cells)))

                    # 2️⃣ Save it (optional) and start the executor thread
                    navigation.save_route_to_file(navigation.pending_route)
                    threading.Thread(
                        target=_execute_route,
                        args=(navigation.pending_route, stop_event),
                        daemon=True
                    ).start()
                else:
                    print("Route computation returned no waypoints.")

        # Display frame if available
        if frame is not None:
            # ─── Update FPS once per second ────────────────────────────────────
            frame_counter += 1
            now = time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps = frame_counter / elapsed
                frame_counter = 0
                fps_timer = now

            # ─── Draw FPS onto the frame ───────────────────────────────────────
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                frame, fps_text,
                org=(10, 30),  # top-left corner
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2
            )

            cv2.imshow("Live Object Detection", frame)

    # Cleanup
    cv2.destroyAllWindows()
    print("display_frames exiting")


def _execute_route(route_cm, stop_event):
    """
    For each (x,y) in route_cm:
      1) send_goto(x,y)
      2) wait until detection.planner.robot_position_cm is within ARRIVAL_THRESHOLD_CM
      3) abort or skip if stop_event is set or timeout expires
    Then send_deliver().
    """
    for x_target, y_target in route_cm:
        # Check for global abort
        if stop_event.is_set():
            print("Route aborted by user")
            return

        # Send the drive command
        robot_comm.send_goto(x_target, y_target)
        start_t = time.time()

        # Poll until arrival, abort, or timeout
        while True:
            # Abort on stop_event
            if stop_event.is_set():
                print("Route aborted during segment")
                return

            # Timeout handling
            elapsed = time.time() - start_t
            if elapsed > config.MAX_SEGMENT_TIME:
                print(f"⚠️ Timeout ({elapsed:.1f}s) to reach ({x_target:.1f},{y_target:.1f}); skipping")
                break

            # Check current pose
            pos = detection.planner.robot_position_cm
            if pos is not None:
                cur_x, cur_y = pos
                dist = math.hypot(x_target - cur_x, y_target - cur_y)
                if dist < config.ARRIVAL_THRESHOLD_CM:
                    print(f"Arrived at ({x_target:.1f}, {y_target:.1f}) → d={dist:.1f}cm in {elapsed:.1f}s")
                    break

            time.sleep(0.05)  # adjust polling rate

    # Final deliver
    robot_comm.send_deliver()
    print("🏁 Route complete, delivered")
