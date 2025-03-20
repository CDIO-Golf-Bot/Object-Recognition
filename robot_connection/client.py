import socket

def send_command(command):
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        # Connect to the server
        host = "10.135.97.57"  # Replace with your EV3's IP address
        port = 12345
        client_socket.connect((host, port))

        # Send the command
        client_socket.send(command.encode())

        # Receive a response from the server
        response = client_socket.recv(1024).decode()
        print("Received: {}".format(response))
    except ConnectionRefusedError:
        print("Could not connect to the server. Is it running?")
    except ConnectionResetError:
        print("Server closed the connection unexpectedly.")
    finally:
        # Close the connection
        client_socket.close()

# Main loop
while True:
    command = input("Enter command (forward/back/pause/run/reverse/stop/quit): ").strip().lower()
    if command == "quit":
        break
    send_command(command)