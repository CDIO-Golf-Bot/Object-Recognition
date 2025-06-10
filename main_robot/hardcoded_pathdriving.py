#!/usr/bin/env python3
from ev3dev2.motor import MoveTank, Motor, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedPercent
from ev3dev2.sensor.lego import GyroSensor
from ev3dev2.sensor import INPUT_3
import time
import math

# --- Constants ---
WHEEL_DIAMETER_CM = 4.0  # Adjust this to your robot's wheel diameter
WHEEL_CIRCUMFERENCE_CM = WHEEL_DIAMETER_CM * math.pi

# --- Initialization ---
tank = MoveTank(OUTPUT_B, OUTPUT_C)
motor_d = Motor(OUTPUT_D)
gyro = GyroSensor(INPUT_3)

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

def drive_distance(distance_cm, speed_pct=30, target_angle=None):
    """
    Drive straight for specified distance in cm, maintaining heading with gyro + PID.
    If target_angle is None, uses current gyro angle as target.
    """
    if target_angle is None:
        target_angle = gyro.angle
    
    # Calculate required motor degrees to travel distance_cm
    degrees = (distance_cm / WHEEL_CIRCUMFERENCE_CM) * 360
    
    # Reset motor positions
    tank.left_motor.position = 0
    tank.right_motor.position = 0
    
    integral = 0
    last_error = 0
    
    while True:
        # Get current motor positions
        left_pos = abs(tank.left_motor.position)
        right_pos = abs(tank.right_motor.position)
        avg_pos = (left_pos + right_pos) / 2
        
        # Check if we've traveled far enough
        if avg_pos >= degrees:
            break
            
        # Gyro correction
        error = target_angle - gyro.angle
        if abs(error) < 1:
            error = 0
        integral += error
        derivative = error - last_error
        last_error = error

        corr = kp * error + ki * integral + kd * derivative
        corr = max(min(corr, max_correction), -max_correction)

        left_spd = max(min(speed_pct - corr, 100), -100)
        right_spd = max(min(speed_pct + corr, 100), -100)

        tank.on(SpeedPercent(left_spd), SpeedPercent(right_spd))
        time.sleep(0.01)
    
    tank.off()

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
        
        # 1) Forward 20 cm
        print("Forward 20 cm")
        drive_distance(20, speed_pct=30)
        
        # 2) Turn right 90°
        print("Turn right 90 degrees")
        turn(90, turn_speed=30)
        
        # 3) Forward 30 cm
        print("Forward 30 cm")
        drive_distance(30, speed_pct=30, target_angle=gyro.angle)
        
        # 4) Turn left 90°
        print("Turn left 90 degrees")
        turn(-90, turn_speed=30)
        
        # 5) Forward 20 cm
        print("Forward 20 cm")
        drive_distance(20, speed_pct=30, target_angle=gyro.angle)
        
        # 6) Reverse motor D
        motor_d.on_for_seconds(-15, 8)
        
        print("Done")
    
    except Exception as e:
        print("Error:", e)
        tank.off()
        motor_d.off()