import time
from ev3dev2.motor import MoveTank, Motor, OUTPUT_B, OUTPUT_C, OUTPUT_D
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_4
from utils import normalize_angle

# ————————————————————————
# ACTUATORS & SENSORS
# ————————————————————————
tank      = MoveTank(OUTPUT_B, OUTPUT_C)
aux_motor = Motor(OUTPUT_D)
gyro      = GyroSensor(INPUT_4)

gyro.mode = 'GYRO-ANG'      # or 'GYRO-ANG-RATE' depending on your firmware
gyro.reset()                # zero the angle counter
time.sleep(0.2)  

# global offset to fuse client heading with raw gyro
gyro_offset = 0.0

def get_heading():
    """Returns current gyro angle (0–360) after offset."""
    raw = gyro.angle
    fused = (normalize_angle(-raw) + gyro_offset) % 360.0
    print("[get_heading] raw={:.1f}, fused={:.1f}".format(raw, fused))
    return fused

def calibrate_gyro_aruco(aruco_heading):
    global gyro_offset
    gyro.reset()
    time.sleep(0.05)
    # invert the raw so it runs CCW-positive
    raw = (-gyro.angle) % 360
    gyro_offset = (aruco_heading - raw) % 360
    fused = get_heading()
    print(" Offset={:.1f} deg, fused heading={:.1f} deg".format(gyro_offset, fused))



