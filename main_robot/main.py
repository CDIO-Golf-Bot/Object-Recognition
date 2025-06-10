#!/usr/bin/env python3

import socket
import threading
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_3
from ev3dev2.motor import Motor, OUTPUT_D, OUTPUT_B, OUTPUT_C, MoveTank, SpeedPercent
import time

# Initialize motors
motor_d = Motor(OUTPUT_D)  # Collect/Deliver motor
tank_drive = MoveTank(OUTPUT_B, OUTPUT_C)  # Driving motors
left_motor = tank_drive.left_motor
right_motor = tank_drive.right_motor

# Initialize gyro
gyro = GyroSensor(INPUT_3)

print("Calibrating gyro, keep robot still...")
time.sleep(1)
gyro.reset()
time.sleep(0.5)
gyro.mode = 'GYRO-ANG'
time.sleep(0.5)

# PID constants for smooth gyro driving
kp = 2.5
ki = 0.0002
kd = 5.5
max_correction = 8
target_angle = 0

def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

def gyro_drive_forward(duration=2.0, base_speed=30):
    """Drive forward for `duration` seconds while using gyro to correct heading."""
    integral = 0
    last_error = 0
    end_time = time.time() + duration

    while time.time() < end_time:
        error = target_angle - gyro.angle
        if abs(error) < 1:  # Deadband to ignore noise
            error = 0
        integral += error
        derivative = error - last_error
        correction = kp * error + ki * integral + kd * derivative
        last_error = error

        correction = clamp(correction, -max_correction, max_correction)

        left_speed = clamp(base_speed - correction, -100, 100)
        right_speed = clamp(base_speed + correction, -100, 100)

        left_motor.on(SpeedPercent(left_speed))
        right_motor.on(SpeedPercent(right_speed))

        time.sleep(0.01)

    left_motor.off()
    right_motor.off()

# Flags for collect and deliver operations
collecting = False
delivering = False
motor_thread = None

def collect():
    global collecting
    print("Starting collection...")
    while collecting:
        motor_d.on(25)
        time.sleep(0.1)
    motor_d.off()
    print("Collection stopped.")

def deliver():
    global delivering
    print("Starting delivery...")
    while delivering:
        motor_d.on(-25)
        time.sleep(0.1)
    motor_d.off()
    print("Delivery stopped.")

def handle_motor(command):
    global collecting, delivering, motor_thread

    if command == "collect":
        if not collecting:
            collecting = True
            delivering = False
            motor_thread = threading.Thread(target=collect, daemon=True)
            motor_thread.start()
        print("Already collecting...")

    elif command == "deliver":
        if not delivering:
            delivering = True
            collecting = False
            motor_thread = threading.Thread(target=deliver, daemon=True)
            motor_thread.start()
        print("Already delivering...")

    elif command == "pause":
        print("Pausing collection/delivery...")
        collecting = False
        delivering = False
        if motor_thread:
            motor_thread.join()
        motor_d.off()

    elif command == "forward":
        print("Gyro-stabilized forward drive...")
        gyro_drive_forward(duration=1, base_speed=30)  # 2s of smooth forward motion

    elif command == "back":
        print("Moving backward...")
        tank_drive.on(-75, -75)

    elif command == "right":
        print("Turning right...")
        tank_drive.on(-50, 50)
        global target_angle
        time.sleep(0.3)
        target_angle = gyro.angle

    elif command == "left":
        print("Turning left...")
        tank_drive.on(50, -50)
        time.sleep(0.3)
        target_angle = gyro.angle

    elif command == "stop":
        print("Stopping drive motors...")
        tank_drive.off()

# Server setup
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
                break
            print("Received: {}".format(data))

            if data in ["collect", "deliver", "pause"]:
                handle_motor(data)
                client_socket.send(b"Command received")
            elif data in ["forward", "back", "left", "right", "stop"]:
                handle_motor(data)
                client_socket.send(b"Command received")
            else:
                client_socket.send(b"Unknown command")
    except ConnectionResetError:
        print("Client disconnected.")
    finally:
        client_socket.close()
        print("Connection closed.")
