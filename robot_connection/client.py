import socket
import keyboard

# EV3 IP and Port
HOST = "10.41.178.57"  # Replace with your EV3's IP address
PORT = 12345

# Create a persistent connection
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    client_socket.connect((HOST, PORT))
    print("Connected to EV3. Use W/A/S/D to move, ↑ for collect, ↓ for deliver, Q to quit.")

    def send_command(command):
        """Sends a movement command to the robot."""
        try:
            client_socket.send(command.encode())
        except (ConnectionResetError, BrokenPipeError):
            print("Lost connection to EV3.")

    def on_key_event(event):
        """Handles key presses and releases."""
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'w':  # Move Forward
                send_command("forward")
            elif event.name == 's':  # Move Backward
                send_command("back")
            elif event.name == 'a':  # Turn right
                send_command("left")
            elif event.name == 'd':  # Turn left
                send_command("right")
            elif event.name == 'c':  # Collect (Previously Reverse)
                send_command("collect")
            elif event.name == 'r':  # Deliver (Previously Run)
                send_command("deliver")
            elif event.name == 'f':  # Deliver (Previously Run)
                send_command("pause")
        elif event.event_type == keyboard.KEY_UP:
            send_command("stop")  # Stop when key is released

    # Listen for keyboard events
    keyboard.hook(on_key_event)
    keyboard.wait('q')  # Press 'q' to quit

except ConnectionRefusedError:
    print("Could not connect to the EV3. Is the server running?")
finally:
    client_socket.close()
    print("Disconnected from EV3.")
