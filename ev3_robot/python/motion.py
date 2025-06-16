#!/usr/bin/env python3
import time, math, json
from ev3dev2.motor import SpeedPercent
import config as config, hardware

# Attempt to initialize Ultrasonic Sensor (port from config)
try:
    from ev3dev2.sensor.lego import UltrasonicSensor
    from ev3dev2.sensor import INPUT_1
    distance_sensor = UltrasonicSensor(INPUT_1)
    ultrasonic_available = True
except Exception:
    print("Warning: Ultrasonic sensor not available.")
    distance_sensor = None
    ultrasonic_available = False

# Shared robot pose for fuse
robot_pose = {
    "x": None,
    "y": None,
    "theta": None,
    "timestamp": time.time()
}

# Internal helpers

def _start_aux():    hardware.aux_motor.on(config.AUX_FORWARD_PCT)

def _stop_aux():     hardware.aux_motor.off()

def _reverse_aux():  
    hardware.aux_motor.on(config.AUX_REVERSE_PCT)
    time.sleep(config.AUX_REVERSE_SEC)
    hardware.aux_motor.off()


def get_corrected_heading():
    if robot_pose["theta"] is not None and time.time() - robot_pose["timestamp"] < 0.5:
        return robot_pose["theta"]
    return hardware.get_heading()


def drive_distance(distance_cm, speed_pct=None, target_angle=None, use_ultrasonic=False):
    """Drive straight for distance_cm using encoders + optional ultrasonic feedback."""
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT
    if target_angle is None:
        target_angle = hardware.get_heading()

    rotations = distance_cm / config.WHEEL_CIRC_CM
    integral = 0.0
    last_error = 0.0

    # If ultrasonic, record initial reading
    if use_ultrasonic and ultrasonic_available:
        start_dist = distance_sensor.distance_centimeters
        print("Ultrasonic start: {:.2f} cm for target {:.2f} cm".format(start_dist, distance_cm))
    else:
        start_dist = None
        print("Ultrasonic sensor data unavailable.")

    _start_aux()
    try:
        start_l = hardware.tank.left_motor.position
        start_r = hardware.tank.right_motor.position

        while True:
            # Check encoder condition
            pos_l = hardware.tank.left_motor.position - start_l
            pos_r = hardware.tank.right_motor.position - start_r
            avg_rot = (abs(pos_l) + abs(pos_r)) / 2.0 / 360.0
            encoder_done = avg_rot >= rotations

            # Check ultrasonic condition
            if use_ultrasonic and ultrasonic_available and start_dist is not None:
                current = distance_sensor.distance_centimeters
                traveled = start_dist - current
                ultra_done = traveled >= distance_cm
            else:
                ultra_done = False

            # Break if either method satisfied
            if encoder_done or ultra_done:
                break

            # PID heading correction
            error = ((target_angle - hardware.get_heading() + 540) % 360) - 180
            if abs(error) < 1.0:
                error = 0.0
            derivative = error - last_error
            last_error = error

            correction = (
                config.GYRO_KP * error +
                0.0 +  # integral disabled
                config.GYRO_KD * derivative
            )
            correction = max(min(correction, config.MAX_CORRECTION), -config.MAX_CORRECTION)

            l_spd = max(min(speed_pct - correction + config.LEFT_BIAS, 100), -100)
            r_spd = max(min(speed_pct + correction, 100), -100)

            hardware.tank.on(SpeedPercent(l_spd), SpeedPercent(r_spd))
            time.sleep(0.01)
    finally:
        hardware.tank.off()
        _stop_aux()

    # Debug output
    print("drive_distance: encoder_rot={:.2f}, target_rot={:.2f}".format(avg_rot, rotations))
    if use_ultrasonic and ultrasonic_available and start_dist is not None:
        end_dist = distance_sensor.distance_centimeters
        print("Ultrasonic end: {:.2f} cm, traveled={:.2f} cm".format(end_dist, (start_dist-end_dist)))



def perform_turn(angle_deg):
    """Rotate in place by angle_deg (±), using gyro feedback."""
    current_heading = get_corrected_heading()
    target = (current_heading + angle_deg) % 360.0
    _start_aux()
    try:
        while abs(((hardware.get_heading() - target + 540) % 360) - 180) > config.ANGLE_TOLERANCE:
            err = ((target - hardware.get_heading() + 540) % 360) - 180
            power = max(min(abs(err) * 0.4, config.TURN_SPEED_PCT), 5)
            if err > 0:
                hardware.tank.on(-power, power)
            else:
                hardware.tank.on(power, -power)
            time.sleep(0.01)
        hardware.tank.off()
        time.sleep(0.1)
    finally:
        hardware.tank.off()
        _stop_aux()


def follow_path(points, start_heading_deg):
    """Turn & drive each leg, passing exact segment heading into drive_distance."""
    hardware.gyro_offset = start_heading_deg
    hardware.calibrate_gyro()
    print("Gyro offset={:.1f} (gyro={})".format(hardware.gyro_offset, hardware.gyro.angle))

    cur_heading = get_corrected_heading()
    cur_x, cur_y = points[0]

    for nx, ny in points[1:]:
        dx = (nx - cur_x) * config.CELL_SIZE_CM
        dy = (ny - cur_y) * config.CELL_SIZE_CM

        # Desired direction to next point
        target_heading = math.degrees(math.atan2(dy, dx)) % 360.0

        # Correct current heading based on live drift
        live_heading = get_corrected_heading()
        drift_error = ((live_heading - cur_heading + 540) % 360) - 180
        if abs(drift_error) > 5.0:
            print("Drift detected ({} deg), correcting...".format(round(drift_error, 1)))
            perform_turn(drift_error)
            cur_heading = get_corrected_heading()

        # Now compute actual angle difference to next point
        delta = ((target_heading - cur_heading + 180) % 360) - 180
        if abs(delta) > config.ANGLE_TOLERANCE:
            perform_turn(delta)
            cur_heading = get_corrected_heading()

        # Drive to the point
        distance = math.hypot(dx, dy)
        if distance > 0:
            drive_distance(distance, speed_pct=config.DRIVE_SPEED_PCT, target_angle=target_heading)

        # Update current state
        cur_heading = target_heading
        cur_x, cur_y = nx, ny


def pure_pursuit_follow(path, lookahead_cm=15, speed_pct=None):
    """Continuously follow a path using the Pure Pursuit algorithm."""
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT

    # Initialize pose from first path point and gyro
    x, y = path[0]
    hardware.calibrate_gyro()

    def find_lookahead_point(path, x, y, L):
        # Find the closest segment point at distance >= L from (x,y)
        for i in range(len(path)-1):
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            # parametric along segment
            dx, dy = x2-x1, y2-y1
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6:
                continue
            # project current position onto segment
            t = ((x-x1)*dx + (y-y1)*dy) / (seg_len**2)
            t = max(0.0, min(1.0, t))
            # point on segment
            px, py = x1 + t*dx, y1 + t*dy
            if math.hypot(px-x, py-y) >= L:
                return (px, py)
        return path[-1]

    _start_aux()
    try:
        while True:
            # read pose
            heading = math.radians(hardware.get_heading())
            # convert grid to cm coords
            x_cm, y_cm = x*config.CELL_SIZE_CM, y*config.CELL_SIZE_CM

            look_pt = find_lookahead_point([(px*config.CELL_SIZE_CM, py*config.CELL_SIZE_CM) for (px,py) in path],
                                           x_cm, y_cm, lookahead_cm)
            # desired heading
            desired_heading = math.degrees(math.atan2(look_pt[1]-y_cm, look_pt[0]-x_cm))

            # PID steer toward lookahead
            error = ((desired_heading - hardware.get_heading() + 540) % 360) - 180
            derivative = error  # simplistic derivative
            corr = (config.GYRO_KP*error + config.GYRO_KD*derivative)
            corr = max(min(corr, config.MAX_CORRECTION), -config.MAX_CORRECTION)

            l_spd = max(min(speed_pct - corr + config.LEFT_BIAS, 100), -100)
            r_spd = max(min(speed_pct + corr, 100), -100)

            hardware.tank.on(SpeedPercent(l_spd), SpeedPercent(r_spd))
            time.sleep(0.01)

            # update pose using odometry
            # simple forward integration
            d = speed_pct/100 * config.WHEEL_CIRC_CM * 0.01
            x_cm += d * math.cos(heading)
            y_cm += d * math.sin(heading)
            x, y = x_cm/config.CELL_SIZE_CM, y_cm/config.CELL_SIZE_CM

    finally:
        hardware.tank.off()
        _stop_aux()


def handle_command(cmd, buf):
    """Process a single JSON command from the network."""
    if 'turn' in cmd:
        if buf['distance_buffer'] > 0:
            drive_distance(buf['distance_buffer'])
            buf['distance_buffer'] = 0
        perform_turn(float(cmd['turn']))

    if 'distance' in cmd:
        buf['distance_buffer'] += float(cmd['distance'])

    if cmd.get('deliver'):
        print("DELIVER".format())
        _reverse_aux()
