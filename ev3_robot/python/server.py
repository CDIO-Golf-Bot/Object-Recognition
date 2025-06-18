# server.py

import socket, json, time, threading
from queue import Queue, Empty

import config, hardware, motion

def handle_client(conn, addr):
    print("Client connected: {}".format(addr))
    state = {'distance_buffer': 0.0}
    robot_pose_queue = Queue()

    # Receiver thread: read lines, handle poses directly, enqueue other commands
    def recv_loop():
        buf = b''
        try:
            while True:
                data = conn.recv(1024)
                if not data:
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
                        motion.robot_pose.update({
                            'x':         float(cmd['pose']['x']),
                            'y':         float(cmd['pose']['y']),
                            'theta':     float(cmd['pose']['theta']),
                            'timestamp': time.time()
                        })
                        #print( "Pose recv: {x:.1f},{y:.1f} (age {age:.2f}s)".format( x=motion.robot_pose['x'], y=motion.robot_pose['y'], age=time.time() - motion.robot_pose['timestamp']))
                        continue

                    # Enqueue non-pose commands
                    robot_pose_queue.put(cmd)
        except Exception as e:
            print("Receiver thread error: {}".format(e))
        finally:
            conn.close()

    threading.Thread(target=recv_loop, daemon=True).start()

    # Main dispatch loop: pull non-pose commands off the queue
    while True:
        try:
            cmd = robot_pose_queue.get(timeout=0.1)
        except Empty:
            continue
        # Full-path command
        if 'path' in cmd and 'heading' in cmd:
            try:
                motion.follow_path(cmd['path'], cmd['heading'])
            except Exception as e:
                print("Exception in follow_path: {}".format(e))

        # Other commands (turn, distance, goto, face, deliver)
        else:
            try:
                motion.handle_command(cmd, state)
            except Exception as e:
                print("Exception in handle_command({}): {}".format(cmd, e))


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
