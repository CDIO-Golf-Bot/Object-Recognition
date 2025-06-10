#!/usr/bin/env python3

import socket
import threading
import time
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_3
from ev3dev2.motor import Motor, OUTPUT_B, OUTPUT_C, OUTPUT_D, MoveTank

# Initialize motors and gyro
motor_d = Motor(OUTPUT_D)
tank_drive = MoveTank(OUTPUT_B, OUTPUT_C)
gyro = GyroSensor(INPUT_3)

# Gyro calibration
print("Calibrating gyro...")
gyro.reset()
time.sleep(1)
gyro.mode = 'GYRO-ANG'
time.sleep(0.5)

# Control flags
collecting = False
delivering = False
stop_driving = False

def collect():
    global collecting
    print("Collecting...")
    while collecting:
        motor_d.on(25)
        time.sleep(0.1)
    motor_d.off()

def deliver():
    global delivering
    print("Delivering...")
    while delivering:
        motor_d.on(-25)
        time.sleep(0.1)
    motor_d.off()

def drive_straight(speed=75, duration=None):
    global stop_driving
    print("Driving straight...")
    target_angle = gyro.angle
    start_time = time.time()
    stop_driving = False

    while not stop_driving:
        error = gyro.angle - target_angle
        correction = max(min(error * 0.6, 20), -20)

        left = max(min(speed - correction, 100), -100)
        right = max(min(speed + correction, 100), -100)
        tank_drive.on(left, right)

        if duration and (time.time() - start_time) >= duration:
            break
        time.sleep(0.05)

    tank_drive.off()
    print("Stopped driving.")

def turn(angle):
    """Turns the robot by a specific angle (positive = right, negative = left)."""
    print("Turning {} degrees...".format(angle))
    gyro.mode = 'GYRO-ANG'
    start_angle = gyro.angle
    target = start_angle + angle
    if angle > 0:
        tank_drive.on(-30, 30)
        while gyro.angle < target:
            time.sleep(0.01)
    else:
        tank_drive.on(30, -30)
        while gyro.angle > target:
            time.sleep(0.01)
    tank_drive.off()

def stop_all():
    global collecting, delivering, stop_driving
    print("Stopping everything...")
    collecting = False
    delivering = False
    stop_driving = True
    motor_d.off()
    tank_drive.off()

def handle_command(cmd):
    global collecting, delivering, stop_driving

    if cmd == "collect":
        if not collecting:
            collecting = True
            delivering = False
            threading.Thread(target=collect, daemon=True).start()

    elif cmd == "deliver":
        if not delivering:
            delivering = True
            collecting = False
            threading.Thread(target=deliver, daemon=True).start()

    elif cmd == "pause":
        stop_all()

    elif cmd.startswith("forward"):
        stop_all()
        parts = cmd.split()
        duration = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        threading.Thread(target=drive_straight, args=(75, duration), daemon=True).start()

    elif cmd == "back":
        stop_all()
        tank_drive.on(-75, -75)

    elif cmd == "left":
        stop_all()
        turn(-88)

    elif cmd == "right":
        stop_all()
        turn(88)

    elif cmd == "stop":
        stop_all()

# Socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("0.0.0.0", 12345))
server_socket.listen(1)
print("Waiting for connection...")

while True:
    client_socket, addr = server_socket.accept()
    print("Connected to {}".format(addr))

    try:
        while True:
            data = client_socket.recv(1024).decode().strip().lower()
            if not data:
                break
            print("Received: {}".format(data))
            handle_command(data)
            client_socket.send(b"OK")
    except (ConnectionResetError, BrokenPipeError):
        print("Client disconnected.")
    finally:
        client_socket.close()
