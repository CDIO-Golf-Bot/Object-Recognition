#!/usr/bin/env python3
import socket
import threading
from ev3dev2.motor import Motor, OUTPUT_D, OUTPUT_B, OUTPUT_C, MoveTank
import time
from queue import Queue

# Initialize motors
motor_d = Motor(OUTPUT_D)  # Collect/Deliver motor
tank_drive = MoveTank(OUTPUT_B, OUTPUT_C)  # Driving motors

# Flag to check if a command is in progress
command_in_progress = False

def handle_motor(command):
    """Executes motor actions based on received commands."""
    global command_in_progress

    # If a command is already in progress, ignore the new command
    if command_in_progress:
        print("Command already in progress, ignoring new command")
        return
    
    # Set flag to indicate a command is in progress
    command_in_progress = True

    if command == "forward":
        print("Moving forward...")
        tank_drive.on_for_seconds(50, 50, 2)  # Adjust duration if necessary
    elif command == "right":
        print("Turning right...")
        tank_drive.on_for_degrees(-30, 30, 600)  # Example only
    elif command == "left":
        print("Turning left...")
        tank_drive.on_for_degrees(30, -30, 600)  # Example only
    elif command == "turn around":
        print("Turning around...")
        tank_drive.on_for_seconds(30, -30, 4)  # Adjust duration if necessary
    elif command == "stop":
        print("Stopping drive motors...")
        tank_drive.off()
    else:
        print("Unknown command: {}".format(command))

    # After completing the action, reset the flag
    command_in_progress = False

# Create a queue for handling motor commands
command_queue = Queue()

def command_executor():
    """Thread function that processes commands asynchronously."""
    while True:
        command = command_queue.get()  # Wait for a command to process
        if command is None:  # Exit the loop if None is received (used to stop the thread)
            break
        handle_motor(command)
        command_queue.task_done()  # Signal that the task is done

# Start the command executor thread
thread = threading.Thread(target=command_executor)
thread.start()

# Create a socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 12345))
server_socket.listen(5)
print("Server listening...")

while True:
    client_socket, addr = server_socket.accept()
    print("Connection from {} established!".format(addr))
    
    try:
        while True:
            data = client_socket.recv(1024).decode().strip().lower()
            if not data:
                break  # Exit loop if client disconnects
            print("Received: {}".format(data))
            
            # Add the command to the queue for processing by the executor thread
            command_queue.put(data)
            client_socket.send(b"Command received")
    except ConnectionResetError:
        print("Client disconnected.")
    finally:
        client_socket.close()
        print("Connection closed.")

# Stop the command executor thread when the server is shut down
command_queue.put(None)  # Send None to stop the thread
thread.join()  # Wait for the thread to finish
