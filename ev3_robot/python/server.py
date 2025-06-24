import socket, json, time, threading, logging
from queue import Queue, Empty

import config, hardware, motion

def poses_are_equal(p1, p2, tol=1e-3):
    return (abs(p1['x'] - p2['x']) < tol and
            abs(p1['y'] - p2['y']) < tol and
            abs(p1['theta'] - p2['theta']) < tol)

def handle_client(conn, addr):
    print("Client connected: {}".format(addr))
    state = {'distance_buffer': 0.0}
    robot_pose_queue = Queue()

    last_pose = {'x': None, 'y': None, 'theta': None}
    last_pose_change_time = time.time()
    client_alive = True

    def recv_loop():
        nonlocal last_pose, last_pose_change_time, client_alive
        buf = b''
        try:
            while True:
                data = conn.recv(1024)
                if not data:
                    print("Client disconnected.")
                    client_alive = False
                    break
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    if not line:
                        continue
                    try:
                        cmd = json.loads(line.decode())
                    except json.JSONDecodeError:
                        print("Bad JSON: {}".format(line))
                        continue

                    # Handle pose updates immediately
                    if 'pose' in cmd:
                        pose = cmd['pose']
                        new_pose = {
                            'x': float(pose['x']),
                            'y': float(pose['y']),
                            'theta': float(pose['theta'])
                        }
                        if last_pose['x'] is not None and poses_are_equal(new_pose, last_pose):
                            pass
                        else:
                            last_pose = new_pose
                            last_pose_change_time = time.time()
                        motion.robot_pose.update({
                            'x': new_pose['x'],
                            'y': new_pose['y'],
                            'theta': new_pose['theta'],
                            'timestamp': last_pose_change_time
                        })
                        continue

                    print("Enqueueing command: {}".format(cmd))
                    robot_pose_queue.put(cmd)
        except Exception as e:
            print("Receiver thread error: {}".format(e))
        finally:
            client_alive = False
            conn.close()

    threading.Thread(target=recv_loop, daemon=True).start()

    while client_alive:
        try:
            cmd = robot_pose_queue.get(timeout=0.1)
        except Empty:
            continue
        if 'path' in cmd and 'heading' in cmd:
            try:
                motion.follow_path(cmd['path'], cmd['heading'])
            except Exception as e:
                print("Exception in follow_path: {}".format(e))
        else:
            try:
                motion.handle_command(cmd, state)
            except Exception as e:
                print("Exception in handle_command({}): {}".format(cmd, e))
    print("handle_client exiting for {}".format(addr))

def run_server(host='', port=12345):
    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print("Listening on {}:{}".format(host or '0.0.0.0', port))

        while True:
            try:
                conn, addr = srv.accept()
                handle_client(conn, addr)
            except Exception as e:
                print("Server error: {}".format(e))
                time.sleep(1)

if __name__ == '__main__':
    try:
        run_server()
    except KeyboardInterrupt:
        print("Server shutting down")