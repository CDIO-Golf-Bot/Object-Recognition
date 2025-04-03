#!/usr/bin/env python3

import socket
import threading
from ev3dev2.motor import Motor, OUTPUT_D, OUTPUT_B, OUTPUT_C, MoveTank
import time

# Initialize motors
motor_d = Motor(OUTPUT_D)  # Collect/Deliver motor
tank_drive = MoveTank(OUTPUT_B, OUTPUT_C)  # Driving motors

# Flags for collect and deliver operations
collecting = False
delivering = False
motor_thread = None  # Track the active motor thread


def collect():
    """Runs the motor to collect while the flag is True."""
    global collecting
    print("Starting collection...")
    while collecting:
        motor_d.on(25)
        time.sleep(0.1)  # Prevent excessive CPU usage
    motor_d.off()
    print("Collection stopped.")


def deliver():
    """Runs the motor to deliver while the flag is True."""
    global delivering
    print("Starting delivery...")
    while delivering:
        motor_d.on(-25)
        time.sleep(0.1)
    motor_d.off()
    print("Delivery stopped.")


def handle_motor(command):
    """Executes motor actions based on received commands."""
    global collecting, delivering, motor_thread

    if command == "collect":
        if not collecting:
            collecting = True
            delivering = False  # Ensure only one action runs at a time
            motor_thread = threading.Thread(target=collect, daemon=True)
            motor_thread.start()
        print("Already collecting...")

    elif command == "deliver":
        if not delivering:
            delivering = True
            collecting = False  # Ensure only one action runs at a time
            motor_thread = threading.Thread(target=deliver, daemon=True)
            motor_thread.start()
        print("Already delivering...")

    elif command == "pause":
        print("Pausing collection/delivery...")
        collecting = False
        delivering = False
        if motor_thread:
            motor_thread.join()  # Ensure the thread finishes
        motor_d.off()

    elif command == "forward":
        print("Moving forward...")
        tank_drive.on(75, 75)

    elif command == "back":
        print("Moving backward...")
        tank_drive.on(-75, -75)

    elif command == "right":
        print("Turning right...")
        tank_drive.on(-50, 50)

    elif command == "left":
        print("Turning left...")
        tank_drive.on(50, -50)

    elif command == "stop":
        print("Stopping drive motors...")
        tank_drive.off()


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

            if data in ["collect", "deliver", "pause"]:
                handle_motor(data)  # Run in a thread only for long-running tasks
                client_socket.send(b"Command received")
            elif data in ["forward", "back", "left", "right", "stop"]:
                handle_motor(data)  # Execute instantly
                client_socket.send(b"Command received")
            else:
                client_socket.send(b"Unknown command")
    except ConnectionResetError:
        print("Client disconnected.")
    finally:
        client_socket.close()
        print("Connection closed.")
