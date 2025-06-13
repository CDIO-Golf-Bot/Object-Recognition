import cv2
import traceback

from config import (
    CAMERA_INDEX, BUFFER_SIZE, FRAMES_PER_SEC,
    FRAME_WIDTH, FRAME_HEIGHT, SKIP_FRAMES,
    ROBOT_HEADING
)
from robot_comm import send_path
from navigation import (
    click_to_set_corners,
    save_route_to_file,
    pending_route,
    full_grid_path,
    selected_goal
)

def capture_frames(frame_queue, stop_event):
    # Try multiple backends
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF,  "MSMF"),
        (cv2.CAP_ANY,   "ANY")
    ]
    cap = None
    for api, name in backends:
        try:
            tmp = cv2.VideoCapture(CAMERA_INDEX, api)
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
        print(f"❌ Unable to open camera #{CAMERA_INDEX} with any backend. Exiting capture.")
        stop_event.set()
        return

    # Configure capture parameters
    cap.set(cv2.CAP_PROP_BUFFERSIZE, BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FPS, FRAMES_PER_SEC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    cnt = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Frame read failed; retrying...")
                continue

            cnt += 1
            if cnt % SKIP_FRAMES == 0:
                try:
                    frame_queue.put(frame, timeout=0.02)
                except Exception:
                    # queue might be full or closed
                    pass
    except Exception as e:
        print("❌ Unexpected error in capture_frames:")
        traceback.print_exc()
        stop_event.set()
    finally:
        cap.release()
        print("📷 capture_frames exiting")

def display_frames(output_queue, stop_event):
    global selected_goal, pending_route, full_grid_path

    cv2.namedWindow("Live Object Detection")
    cv2.setMouseCallback("Live Object Detection", click_to_set_corners)

    try:
        while not stop_event.is_set():
            try:
                frame = output_queue.get(timeout=0.02)
            except Exception:
                # No frame available — still handle keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    stop_event.set()
                    break
                elif key == ord('1'):
                    selected_goal = 'A'; print("✅ Selected Goal A")
                elif key == ord('2'):
                    selected_goal = 'B'; print("✅ Selected Goal B")
                elif key == ord('s'):
                    if pending_route:
                        save_route_to_file(pending_route)
                    if full_grid_path:
                        send_path(full_grid_path, ROBOT_HEADING)
                continue

            cv2.imshow("Live Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break
            elif key == ord('1'):
                selected_goal = 'A'; print("✅ Selected Goal A")
            elif key == ord('2'):
                selected_goal = 'B'; print("✅ Selected Goal B")
            elif key == ord('s'):
                if pending_route:
                    save_route_to_file(pending_route)
                if full_grid_path:
                    send_path(full_grid_path, ROBOT_HEADING)
    except Exception as e:
        print("❌ Unexpected error in display_frames:")
        traceback.print_exc()
        stop_event.set()
    finally:
        cv2.destroyAllWindows()
        print("🖼️ display_frames exiting")
