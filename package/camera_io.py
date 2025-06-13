import cv2

from config import ROBOT_HEADING
from robot_comm import send_path
from navigation import click_to_set_corners, save_route_to_file, pending_route, full_grid_path, selected_goal
from config import SKIP_FRAMES

def capture_frames(frame_queue, stop_event, cap=None, skip_frames=SKIP_FRAMES):
    import config
    if cap is None:
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, config.BUFFER_SIZE)
        cap.set(cv2.CAP_PROP_FPS, config.FRAMES_PER_SEC)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    cnt = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        cnt += 1
        if cnt % skip_frames:
            try:
                frame_queue.put(frame, timeout=0.02)
            except:
                pass
    cap.release()
    print("📷 capture_frames exiting")

def display_frames(output_queue, stop_event):
    global selected_goal, pending_route, full_grid_path

    cv2.namedWindow("Live Object Detection")
    cv2.setMouseCallback("Live Object Detection", click_to_set_corners)

    while not stop_event.is_set():
        try:
            frame = output_queue.get(timeout=0.02)
        except:
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

    print("🖼️ display_frames exiting")
    cv2.destroyAllWindows()
