#!/usr/bin/env python3

from ev3dev2.motor import Motor, OUTPUT_D
from time import sleep

# Initialize the motor
motor_left = Motor(OUTPUT_D)

# Run the motor
print("Motor connected!")
motor_left.on_for_seconds(20, 20)  # Runs the motor for 3 seconds
motor_left.on_for_seconds(-15, 20)  # Runs the motor for 3 seconds