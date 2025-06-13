#!/usr/bin/env python3
from ev3dev2.motor import MoveTank, Motor, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedPercent
from ev3dev2.sensor.lego import GyroSensor, UltrasonicSensor
from ev3dev2.sensor import INPUT_4, INPUT_1
import time
import math

# --- Constants ---
WHEEL_DIAMETER_CM = 4.13  # Adjust this to your robot's wheel diameter
WHEEL_CIRCUMFERENCE_CM = WHEEL_DIAMETER_CM * math.pi

# --- Initialization ---
tank = MoveTank(OUTPUT_B, OUTPUT_C)
motor_d = Motor(OUTPUT_D)
gyro = GyroSensor(INPUT_4)
try:
    distance_sensor = UltrasonicSensor(INPUT_1)
    ultrasonic_available = True
except Exception as e:
    print("Warning: Ultrasonic sensor not available:", e)
    ultrasonic_available = False


# --- Gyro calibration ---
print("Calibrating gyro, keep robot still...")
gyro.mode = 'GYRO-CAL'  # Start calibration
time.sleep(1)
gyro.mode = 'GYRO-ANG'  # Switch to angle mode
time.sleep(0.1)  # Give it a moment

# --- PID-corrected straight drive by distance ---
# tuned constants from your script:
kp = 2.5
ki = 0.0002
kd = 5.5
max_correction = 8

def drive_distance(distance_cm, speed_pct=30):
    """
    Drive straight for specified distance in cm using on_for_rotations.
    """
    rotations = distance_cm / WHEEL_CIRCUMFERENCE_CM
    print("\n--- Drive Debug Start ---")
    print("Target distance: {:.2f} cm ({} rotations)".format(distance_cm, rotations))

    if ultrasonic_available:
        start_distance = distance_sensor.distance_centimeters
        print("Initial ultrasonic distance: {:.2f} cm".format(start_distance))
    else:
        start_distance = None
        print("Ultrasonic sensor data unavailable.")

    # Perform the movement
    tank.on_for_rotations(SpeedPercent(speed_pct), SpeedPercent(speed_pct), rotations)

    # Debug output
    print("--- Drive Debug End ---")
    if ultrasonic_available and start_distance is not None:
        end_distance = distance_sensor.distance_centimeters
        distance_change = start_distance - end_distance
        print("  Final ultrasonic: {:.2f} cm".format(end_distance))
        print("  Ultrasonic delta: {:.2f} cm\n".format(distance_change))
    else:
        print("  Ultrasonic sensor data unavailable.\n")




# --- Improved gyro-based turn ---
def turn(angle_deg, turn_speed=30):
    """
    Turn in place by angle_deg (positive = right, negative = left).
    Uses gyro feedback for precise turning and stops at the exact angle.
    """
    start_angle = gyro.angle
    target_angle = start_angle + angle_deg
    angle_tolerance = 1.0  # degrees

    # Simple proportional control
    while abs(gyro.angle - target_angle) > angle_tolerance:
        error = target_angle - gyro.angle
        power = min(max(abs(error) * 0.3, 10), turn_speed)  # Proportional control

        if error > 0:
            tank.on(-power, power)  # Right turn
        else:
            tank.on(power, -power)  # Left turn

        time.sleep(0.01)

    tank.off()
    time.sleep(0.1)  # Brief pause to settle

    # Final correction if needed
    final_error = target_angle - gyro.angle
    if abs(final_error) > angle_tolerance:
        if final_error > 0:
            tank.on(-5, 5)
        else:
            tank.on(5, -5)
        time.sleep(abs(final_error) * 0.01)
        tank.off()

# --- Main sequence ---
if __name__ == '__main__':
    try:
        motor_d.on(40)

        # 1) Forward 100 cm
        print("Forward 100 cm")
        drive_distance(100, speed_pct=30)

        # 6) Reverse motor D
        motor_d.on_for_seconds(-15, 8)

        print("Done")

    except Exception as e:
        print("Error:", e)
        tank.off()
        motor_d.off()
