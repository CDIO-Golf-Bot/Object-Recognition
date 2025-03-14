#!/usr/bin/env python3

from ev3dev2.motor import Motor, OUTPUT_B
from time import sleep

# Initialize the motor
motor_left = Motor(OUTPUT_B)

# Run the motor
print("Motor connected!")
motor_left.on_for_seconds(50, 3)  # Runs the motor for 3 seconds