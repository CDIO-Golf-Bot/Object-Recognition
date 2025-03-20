#!/usr/bin/env python3

import socket
import threading
from ev3dev2.motor import Motor, OUTPUT_D, OUTPUT_B, OUTPUT_C, MoveTank

# Initialize the motors
motor_d = Motor(OUTPUT_D)  # For motor D
tank_drive = MoveTank(OUTPUT_B, OUTPUT_C)  # Use MoveTank for synchronized control of B and C

# Function to handle motor actions
def handle_motor(command):
    if command == "run":
        print("Running motor D forward...")
        motor_d.on_for_seconds(100, 20)  # Run motor D forward at 20% power for 20 seconds
    elif command == "reverse":
        print("Running motor D in reverse...")
        motor_d.on_for_seconds(-15, 20)  # Run motor D in reverse at 15% power for 20 seconds
    elif command == "pause":
        print("Pausing motor D...")
        motor_d.off()  # Stop motor D
    elif command == "forward":
        print("Running motors B and C forward...")
        tank_drive.on_for_seconds(100, 100, 10)  # Move forward at 20% power for 10 seconds
    elif command == "back":
        print("Running motors B and C in reverse...")
        tank_drive.on_for_seconds(-20, -20, 10)  # Move backward at 20% power for 10 seconds
    elif command == "stop":
        print("Stopping motors B and C...")
        tank_drive.off()  # Stop both motors

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific address and port
host = "0.0.0.0"  # Listen on all available interfaces
port = 12345      # Port to listen on
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(1)
print("Server listening on {}:{}...".format(host, port))

while True:
    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print("Connection from {} established!".format(client_address))

    while True:
        try:
            # Receive data from the client
            data = client_socket.recv(1024).decode().strip().lower()
            if not data:
                break  # Exit the loop if the client disconnects
            print("Received command: {}".format(data))

            # Process the command in a separate thread
            if data in ["run", "reverse", "stop", "forward", "back", "pause"]:
                threading.Thread(target=handle_motor, args=(data,)).start()
                response = "Command '{}' executed!".format(data)
            elif data == "quit":
                response = "Closing connection..."
                break  # Exit the loop if the client sends "quit"
            else:
                response = "Unknown command!"

            # Send a response back to the client
            client_socket.send(response.encode())
        except ConnectionResetError:
            print("Client disconnected unexpectedly!")
            break

    # Close the connection
    client_socket.close()
    print("Connection closed.")