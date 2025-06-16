import socket, json, time
import config as config, hardware, motion


def run_server(host='', port=12345):
    """Main TCP loop: accept connections, recv JSON, dispatch to motion."""
    hardware.calibrate_gyro()

    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print("Listening on {}:{}".format(host or '0.0.0.0', port))

        while True:
            try:
                conn, addr = srv.accept()
                print("Client connected: {}".format(addr))
                state = {'distance_buffer': 0.0}
                recv_buffer = b''

                with conn:
                    while True:
                        try:
                            data = conn.recv(1024)
                        except ConnectionResetError:
                            print("Connection reset by peer")
                            break
                        if not data:
                            # flush any pending distance
                            if state['distance_buffer'] > 0:
                                motion.drive_distance(state['distance_buffer'])
                            break

                        recv_buffer += data
                        # process all complete lines
                        while b'\n' in recv_buffer:
                            line, recv_buffer = recv_buffer.split(b'\n', 1)
                            if not line:
                                continue
                            try:
                                cmd = json.loads(line.decode())
                            except json.JSONDecodeError:
                                print("Bad JSON: {}".format(line))
                                continue

                            # Pose update
                            if 'pose' in cmd:
                                motion.robot_pose.update({
                                    'x': cmd['pose']['x'],
                                    'y': cmd['pose']['y'],
                                    'theta': cmd['pose']['theta'],
                                    'timestamp': time.time()
                                })

                            # Legacy full-path command
                            elif 'path' in cmd and 'heading' in cmd:
                                motion.follow_path(cmd['path'], cmd['heading'])

                            # Incremental: turn or distance
                            else:
                                motion.handle_command(cmd, state)
            except Exception as e:
                print("Server error: {}".format(e))
                time.sleep(1)


if __name__ == '__main__':
    try:
        run_server()
    except KeyboardInterrupt:
        print("Server shutting down")
