import cv2
import traceback

from robot_client import config, robot_comm, navigation
from robot_client.navigation.planner import compute_best_route
from robot_client import detection

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
                print(f"✅ Camera opened with {name} backend")
                cap = tmp
                break
            else:
                tmp.release()
                print(f"⚠️ Could not open camera with {name} backend")
        except Exception as e:
            print(f"⚠️ Exception using {name} backend: {e}")

    if cap is None:
        print(f"❌ Unable to open camera #{config.CAMERA_INDEX} with any backend. Exiting capture.")
        stop_event.set()
        return

    # Configure capture parameters
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   config.BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FPS,          config.FRAMES_PER_SEC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    cnt = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame read failed; retrying...")
                continue

            cnt += 1
            if cnt % config.SKIP_FRAMES == 0:
                try:
                    frame_queue.put(frame, timeout=0.02)
                except Exception:
                    pass
    except Exception:
        print("❌ Unexpected error in capture_frames:")
        traceback.print_exc()
        stop_event.set()
    finally:
        cap.release()
        print("📷 capture_frames exiting")


def display_frames(output_queue, stop_event):
    """Show frames, handle keypresses & mouse clicks, plan & send routes on 's'."""
    cv2.namedWindow("Live Object Detection")
    cv2.setMouseCallback("Live Object Detection", navigation.click_to_set_corners)

    try:
        while not stop_event.is_set():
            try:
                frame = output_queue.get(timeout=0.02)
            except Exception:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_event.set()
                    break
                elif key == ord('1'):
                    navigation.selected_goal = 'A'
                    print("✅ Selected Goal A")
                elif key == ord('2'):
                    navigation.selected_goal = 'B'
                    print("✅ Selected Goal B")
                elif key == ord('s'):
                    # Plan a fresh route based on detected balls
                    if detection.ball_positions_cm:
                        route_cm, grid_cells = compute_best_route(
                            detection.ball_positions_cm,
                            navigation.selected_goal
                        )
                        print(f"⚠️  Planned route: {len(route_cm)} waypoints, {len(grid_cells)} grid points")
                        navigation.pending_route = route_cm
                        navigation.full_grid_path = grid_cells
                    else:
                        print("⚠️  No balls detected; cannot plan route")

                    # Save route and send path if available
                    if navigation.pending_route:
                        navigation.save_route_to_file(navigation.pending_route)
                    if navigation.full_grid_path:
                        robot_comm.send_path(navigation.full_grid_path, config.ROBOT_HEADING)
                continue

            cv2.imshow("Live Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break
            elif key == ord('1'):
                navigation.selected_goal = 'A'
                print("✅ Selected Goal A")
            elif key == ord('2'):
                navigation.selected_goal = 'B'
                print("✅ Selected Goal B")
            elif key == ord('s'):
                # Plan a fresh route based on detected balls
                if detection.ball_positions_cm:
                    route_cm, grid_cells = compute_best_route(
                        detection.ball_positions_cm,
                        navigation.selected_goal
                    )
                    print(f"⚠️  Planned route: {len(route_cm)} waypoints, {len(grid_cells)} grid points")
                else:
                    print("⚠️  No balls detected; cannot plan route")

                # Save route and send path if available
                if navigation.pending_route:
                    navigation.save_route_to_file(navigation.pending_route)
                if navigation.full_grid_path:
                    robot_comm.send_path(navigation.full_grid_path, config.ROBOT_HEADING)
    except Exception:
        print("❌ Unexpected error in display_frames:")
        traceback.print_exc()
        stop_event.set()
    finally:
        cv2.destroyAllWindows()
        print("🖼️ display_frames exiting")
