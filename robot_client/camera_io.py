import threading
import cv2
import traceback
import math
import time
from queue import Empty

from robot_client import config, robot_comm, navigation, detection
from robot_client.navigation import planner, grid_utils


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
def plan_and_execute_next_route(stop_event):
    balls = detection.ball_positions_cm
    if not balls:
        print("No balls detected; cannot plan next route.")
        return
    route_cm, grid_cells = planner.compute_best_route(
        balls,
        navigation.selected_goal
    )
    if route_cm:
        turn_points = navigation.compress_path(grid_cells)
        turn_points_cm = navigation.grid_path_to_cm(turn_points)
        robot_pos = planner.robot_position_cm
        if robot_pos is not None and len(turn_points_cm) > 1:
            dist0 = math.hypot(turn_points_cm[0][0] - robot_pos[0], turn_points_cm[0][1] - robot_pos[1])
            if dist0 < config.ARRIVAL_THRESHOLD_CM:
                waypoints_to_send = turn_points_cm[1:]
            else:
                waypoints_to_send = turn_points_cm
        else:
            waypoints_to_send = turn_points_cm

        navigation.pending_route = turn_points_cm
        navigation.full_grid_path = grid_cells
        print(f"Planned next route: {len(waypoints_to_send)} waypoints")
        navigation.save_route_to_file(waypoints_to_send)
        threading.Thread(
            target=_execute_route,
            args=(waypoints_to_send, stop_event, lambda: plan_and_execute_next_route(stop_event)),
            daemon=True
        ).start()
    else:
        print("No more balls to collect.")

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
                route_cm, grid_cells = planner.compute_best_route(
                    balls,
                    navigation.selected_goal
                )
                if route_cm:
                    turn_points = navigation.compress_path(grid_cells)
                    turn_points_cm = navigation.grid_path_to_cm(turn_points)

                    # Exclude robot's current position from waypoints to send
                    robot_pos = planner.robot_position_cm
                    if robot_pos is not None and len(turn_points_cm) > 1:
                        # If the first turn point is very close to the robot, skip it
                        dist0 = math.hypot(turn_points_cm[0][0] - robot_pos[0], turn_points_cm[0][1] - robot_pos[1])
                        if dist0 < config.ARRIVAL_THRESHOLD_CM:
                            waypoints_to_send = turn_points_cm[1:]
                        else:
                            waypoints_to_send = turn_points_cm
                    else:
                        waypoints_to_send = turn_points_cm

                    navigation.pending_route = turn_points_cm  # For visualization (draw full path)
                    navigation.full_grid_path = grid_cells
                    print(f"Planned route: {len(waypoints_to_send)} waypoints (sent), {len(turn_points_cm)} turn points (visualized), {len(grid_cells)} grid points")

                    navigation.save_route_to_file(waypoints_to_send)
                    threading.Thread(
                        target=_execute_route,
                        args=(waypoints_to_send, stop_event, lambda: plan_and_execute_next_route(stop_event)),
                        daemon=True
                    ).start()
                else:
                    print("Route computation returned no waypoints.")

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


def _execute_route(route_cm, stop_event, on_complete=None):
    """Drive each segment until arrival, with abort & timeout."""
    for x_target, y_target in route_cm:
        if stop_event.is_set():
            print("Route aborted by user")
            return

        robot_comm.send_goto(x_target, y_target)
        start_t = time.time()
        while True:
            if stop_event.is_set():
                print("Route aborted during segment")
                return
            elapsed = time.time() - start_t
            if elapsed > config.MAX_SEGMENT_TIME:
                print(f"⚠️ Timeout ({elapsed:.1f}s) to reach ({x_target:.1f},{y_target:.1f}); skipping")
                break

            pos = detection.planner.robot_position_cm
            if pos is not None:
                cur_x, cur_y = pos
                dist = math.hypot(x_target - cur_x, y_target - cur_y)
                if dist < config.ARRIVAL_THRESHOLD_CM:
                    print(f"Arrived at ({x_target:.1f}, {y_target:.1f}) → d={dist:.1f}cm in {elapsed:.1f}s")
                    break

            time.sleep(0.05)

    robot_comm.send_deliver()
    print("🏁 Route complete, delivered")
    if on_complete:
        on_complete()