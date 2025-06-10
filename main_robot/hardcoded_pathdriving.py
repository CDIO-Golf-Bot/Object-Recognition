#!/usr/bin/env python3
from ev3dev2.motor import MoveTank, Motor, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedPercent
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_3
import time

# --- Initialization ---
tank = MoveTank(OUTPUT_B, OUTPUT_C)
motor_d = Motor(OUTPUT_D)
gyro = GyroSensor(INPUT_3)

# --- Gyro calibration ---
print("Calibrating gyro, keep robot still...")
gyro.calibrate()            # auto–zero bias
time.sleep(1)               # give it a moment

# --- Simple PID-corrected straight drive ---
# tuned constants from your script:
kp = 2.5
ki = 0.0002
kd = 5.5
max_correction = 8

def drive_forward(duration_s=2.0, speed_pct=30, target_angle=0):
    """Drive straight for duration_s seconds, holding target_angle with gyro + PID."""
    integral = 0
    last_error = 0
    end = time.time() + duration_s
    while time.time() < end:
        error = target_angle - gyro.angle
        if abs(error) < 1:
            error = 0
        integral += error
        derivative = error - last_error
        last_error = error

        corr = kp * error + ki * integral + kd * derivative
        corr = max(min(corr, max_correction), -max_correction)

        left_spd  = max(min(speed_pct - corr, 100), -100)
        right_spd = max(min(speed_pct + corr, 100), -100)

        tank.on(SpeedPercent(left_spd), SpeedPercent(right_spd))
        time.sleep(0.01)
    tank.off()

# --- Simple gyro-based turn ---
def turn(angle_deg, turn_speed=30):
    """
    Turn in place by angle_deg (positive = right, negative = left).
    Uses raw tank on/off and gyro feedback to stop at the right heading.
    """
    start = gyro.angle
    target = start + angle_deg

    # choose rotation direction
    if angle_deg > 0:
        tank.on(-turn_speed, turn_speed)
        # spin until we hit or exceed the target
        while gyro.angle < target:
            time.sleep(0.01)
    else:
        tank.on(turn_speed, -turn_speed)
        while gyro.angle > target:
            time.sleep(0.01)
    tank.off()

# --- Main sequence ---
if __name__ == '__main__':
    motor_d.on(40)
    # 1) Forward 2 s
    print("Forward 2 s")
    drive_forward(duration_s=4.0, speed_pct=30, target_angle=0)

    # 2) Turn left 90°
    print("Turn left 90 decrees")
    # (negative for left)
    turn(90, turn_speed=30)

    # 3) Forward another 2 s
    print("Forward 2 s")
    drive_forward(duration_s=5.0, speed_pct=30, target_angle=gyro.angle)

    turn(-90, turn_speed=30)
    drive_forward(duration_s=4.0, speed_pct=30, target_angle=gyro.angle)

    motor_d.on_for_seconds(-15, 8)

    print("Done")
