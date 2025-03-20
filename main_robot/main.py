#!/usr/bin/env python3

import socket
import threading
from ev3dev2.motor import Motor, OUTPUT_D

# Initialize the motor
motor = Motor(OUTPUT_D)

# Function to handle motor actions
def handle_motor(command):
    if command == "run":
        print("Running motor D forward...")
        motor.on_for_seconds(20, 20)  # Run motor forward at 20% power for 20 seconds
    elif command == "reverse":
        print("Running motor D in reverse...")
        motor.on_for_seconds(-15, 20)  # Run motor in reverse at 15% power for 20 seconds
    elif command == "stop":
        print("Stopping motor D...")
        motor.off()  # Stop the motor

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
            if data in ["run", "reverse", "stop"]:
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