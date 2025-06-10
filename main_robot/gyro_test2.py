#!/usr/bin/env python3

from ev3dev2.motor import LargeMotor, OUTPUT_B, OUTPUT_C, SpeedPercent
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_3
from ev3dev2.sound import Sound
import time

# Init
left_motor = LargeMotor(OUTPUT_B)
right_motor = LargeMotor(OUTPUT_C)
gyro = GyroSensor(INPUT_3)
sound = Sound()

# Calibrate
print("Calibrating gyro, keep robot still...")
gyro.calibrate()
time.sleep(1)

# PID constants
kp = 2.5        # 	Current error	Fast correction
ki = 0.0002      #  	Cumulative past error	Steady drift compensation
kd = 5.5        #  	Speed of error change	Smooths motion
max_correction = 8

# Clamp function
def clamp(val, min_val, max_val):
    return max(min(val, max_val), min_val)

# PID loop
target_angle = 0
integral = 0
last_error = 0
base_speed = 30  # Base forward speed as percent
duration = 15  # seconds
end_time = time.time() + duration

while time.time() < end_time:
    error = target_angle - gyro.angle
    integral += error
    derivative = error - last_error
    correction = kp * error + ki * integral + kd * derivative
    last_error = error

    # Clamp correction to avoid exceeding motor speed limits
    correction = clamp(correction, -max_correction, max_correction)

    # Apply corrected speeds
    left_speed = clamp(base_speed - correction, -100, 100)
    right_speed = clamp(base_speed + correction, -100, 100)

    left_motor.on(SpeedPercent(left_speed))
    right_motor.on(SpeedPercent(right_speed))

    print("Time: {} Gyro: {} L:{:.1f} R:{:.1f}".format(
    int((time.time() - (end_time - duration)) * 1000),
    gyro.angle,
    left_speed,
    right_speed
))
    time.sleep(0.01)

# Stop motors
left_motor.off()
right_motor.off()
