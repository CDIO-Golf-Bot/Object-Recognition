import socket
import json

def run_mock_server(host='0.0.0.0', port=12345):
    print("🧪 Mock robot server listening on {}:{}...".format(host, port))
    
    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)

        while True:
            conn, addr = srv.accept()
            print("🤝 Client connected: {}".format(addr))
            buf = b''

            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        print("🔌 Client disconnected.")
                        break
                    buf += data
                    while b'\n' in buf:
                        line, buf = buf.split(b'\n', 1)
                        try:
                            cmd = json.loads(line.decode())
                            if 'path' in cmd:
                                print(" Received path with {} waypoints".format(len(cmd['path'])))
                            elif 'pose' in cmd:
                                pose = cmd['pose']
                                print(" Pose update: x={:.2f}, y={:.2f}, θ={:.1f}".format(
                                    pose['x'], pose['y'], pose['theta']))
                            elif 'deliver' in cmd:
                                print(" Deliver command received")
                            else:
                                print(" Unknown command: {}".format(cmd))
                        except json.JSONDecodeError:
                            print(" Bad JSON: {}".format(line))

if __name__ == "__main__":
    run_mock_server()
