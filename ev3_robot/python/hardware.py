import time
from ev3dev2.motor import MoveTank, Motor, OUTPUT_B, OUTPUT_C, OUTPUT_D
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_4

# ————————————————————————
# ACTUATORS & SENSORS
# ————————————————————————
tank      = MoveTank(OUTPUT_B, OUTPUT_C)
aux_motor = Motor(OUTPUT_D)
gyro      = GyroSensor(INPUT_4)

# global offset to fuse client heading with raw gyro
gyro_offset = 0.0

def get_heading():
    """Returns gyro heading (0–360) with CCW-positive rotation (matches Aruco)."""
    return (-gyro.angle + gyro_offset) % 360.0

def calibrate_gyro():
    global gyro_offset
    print("Calibrating gyro, keep robot still...")
    gyro.calibrate()    # built-in: enters GYRO-CAL, waits for stability, returns to GYRO-ANG
    gyro.reset()        # zero the angle reading
    gyro_offset = 0.0
    print(" Gyro calibrated and zeroed.")

