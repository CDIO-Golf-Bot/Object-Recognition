import socket

# EV3 IP and Port
HOST = "10.137.48.57"  # Replace with your EV3's IP address
PORT = 12345

# Create a persistent connection
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    client_socket.connect((HOST, PORT))
    print("Connected to EV3.")
    print("Enter W/A/S/D to move, C to collect, R to deliver, F to pause, Q to quit.")

    while True:
        key = input("Enter command: ").lower()

        if key == 'w':
            client_socket.send(b"forward")
        elif key == 's':
            client_socket.send(b"back")
        elif key == 'a':
            client_socket.send(b"left")
        elif key == 'd':
            client_socket.send(b"right")
        elif key == 'c':
            client_socket.send(b"collect")
        elif key == 'r':
            client_socket.send(b"deliver")
        elif key == 'f':
            client_socket.send(b"pause")
        elif key == 'q':
            print("Quitting...")
            break
        else:
            print("Invalid key. Try again.")

        # Optional: tell robot to stop after each command (unless it's collect/deliver)
        if key in ['w', 's', 'a', 'd']:
            client_socket.send(b"stop")

except ConnectionRefusedError:
    print("Could not connect to the EV3. Is the server running?")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    client_socket.close()
    print("Disconnected from EV3.")
