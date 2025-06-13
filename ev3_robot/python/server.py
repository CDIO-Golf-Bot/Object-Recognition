import socket, json, time
import config as config, hardware, motion

def run_server(host='', port=12345):
    """Main TCP loop: accept a connection, recv JSON, dispatch to motion."""
    hardware.calibrate_gyro()

    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        print("Listening on {}:{}…".format(host or '0.0.0.0', port))

        while True:
            conn, addr = srv.accept()
            print("Client connected: {}".format(addr))
            buf =     b''
            state =   {'distance_buffer':0.0}

            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        # flush any leftover distance
                        if state['distance_buffer']>0:
                            motion.drive_distance(state['distance_buffer'])
                        break
                    buf += data
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n',1)
                        try:
                            cmd = json.loads(line.decode())
                            if 'path' in cmd and 'heading' in cmd:
                                motion.follow_path(cmd['path'], cmd['heading'])
                            else:
                                motion.handle_command(cmd, state)
                        except json.JSONDecodeError:
                            print("Bad JSON: {}".format(line))
