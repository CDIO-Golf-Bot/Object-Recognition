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
    # Use vision pose when fresh, otherwise gyro
    if robot_pose["theta"] is not None and time.time() - robot_pose["timestamp"] < 0.5:
        return robot_pose["theta"]
    return hardware.get_heading()


def drive_distance(distance_cm, speed_pct=None, target_angle=None, use_ultrasonic=False):
    """Drive straight for distance_cm using encoders + optional ultrasonic feedback."""
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT
    if target_angle is None:
        target_angle = get_corrected_heading()

    rotations = distance_cm / config.WHEEL_CIRC_CM
    integral = 0.0
    last_error = 0.0

    # If ultrasonic, record initial reading
    if use_ultrasonic and ultrasonic_available:
        start_dist = distance_sensor.distance_centimeters
        print("Ultrasonic start: {:.2f} cm for target {:.2f} cm".format(start_dist, distance_cm))
    else:
        start_dist = None
        if use_ultrasonic:
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
        # Primary gyro-based turn
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

        # Final vision-based micro-correction
        if robot_pose["theta"] is not None and time.time() - robot_pose["timestamp"] < 0.5:
            residual = ((robot_pose["theta"] - hardware.get_heading() + 540) % 360) - 180
            if abs(residual) > config.ANGLE_TOLERANCE:
                print("Vision-based turn-correction: {:.1f} deg".format(residual))
                # slow spin to correct residual
                adjust_power = 5
                if residual > 0:
                    hardware.tank.on(-adjust_power, adjust_power)
                else:
                    hardware.tank.on(adjust_power, -adjust_power)
                time.sleep(abs(residual) / 30.0)
                hardware.tank.off()
    finally:
        hardware.tank.off()
        _stop_aux()


def follow_path(points, start_heading_deg):
    """Turn & drive each leg, fusing vision pose when available."""
    hardware.gyro_offset = start_heading_deg
    hardware.calibrate_gyro()
    print("Gyro offset={:.1f} (gyro={})".format(hardware.gyro_offset, hardware.gyro.angle))

    # Initialize from vision if fresh
    if robot_pose["x"] is not None and time.time() - robot_pose["timestamp"] < 0.5:
        cur_x = robot_pose["x"] / config.CELL_SIZE_CM
        cur_y = robot_pose["y"] / config.CELL_SIZE_CM
        cur_heading = robot_pose["theta"]
    else:
        cur_x, cur_y = points[0]
        cur_heading = get_corrected_heading()

    for nx, ny in points[1:]:
        # Fuse vision at each segment start
        if robot_pose["x"] is not None and time.time() - robot_pose["timestamp"] < 0.5:
            # Update from latest vision
            cur_x = robot_pose["x"] / config.CELL_SIZE_CM
            cur_y = robot_pose["y"] / config.CELL_SIZE_CM
            cur_heading = robot_pose["theta"]

        dx = (nx - cur_x) * config.CELL_SIZE_CM
        dy = (ny - cur_y) * config.CELL_SIZE_CM
        target_heading = math.degrees(math.atan2(dy, dx)) % 360.0

        # Detect and correct large drift
        live_heading = get_corrected_heading()
        drift_error = ((live_heading - cur_heading + 540) % 360) - 180
        if abs(drift_error) > 5.0:
            print("Drift detected ({:.1f} deg), correcting...".format(drift_error))
            perform_turn(drift_error)
            cur_heading = get_corrected_heading()

        # Turn toward next waypoint
        delta = ((target_heading - cur_heading + 180) % 360) - 180
        if abs(delta) > config.ANGLE_TOLERANCE:
            perform_turn(delta)
            cur_heading = get_corrected_heading()

        # Drive straight to waypoint
        distance = math.hypot(dx, dy)
        if distance > 0:
            drive_distance(distance, speed_pct=config.DRIVE_SPEED_PCT, target_angle=target_heading)

        cur_heading = target_heading
        cur_x, cur_y = nx, ny


def pure_pursuit_follow(path, lookahead_cm=15, speed_pct=None):
    """Continuously follow a path using the Pure Pursuit algorithm, fusing vision pose."""
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT

    # Initialize pose from first path point and gyro
    x, y = path[0]
    hardware.calibrate_gyro()

    def find_lookahead_point(path_pts, x, y, L):
        for i in range(len(path_pts)-1):
            x1, y1 = path_pts[i]
            x2, y2 = path_pts[i+1]
            dx, dy = x2-x1, y2-y1
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6:
                continue
            t = ((x-x1)*dx + (y-y1)*dy) / (seg_len**2)
            t = max(0.0, min(1.0, t))
            px, py = x1 + t*dx, y1 + t*dy
            if math.hypot(px-x, py-y) >= L:
                return (px, py)
        return path_pts[-1]

    _start_aux()
    try:
        while True:
            # Grab vision pose if fresh
            if robot_pose["x"] is not None and time.time() - robot_pose["timestamp"] < 0.5:
                x_cm = robot_pose["x"]
                y_cm = robot_pose["y"]
            else:
                # convert grid to cm coords if no fresh vision
                x_cm, y_cm = x * config.CELL_SIZE_CM, y * config.CELL_SIZE_CM

            heading = math.radians(hardware.get_heading())
            look_pt = find_lookahead_point(
                [(px*config.CELL_SIZE_CM, py*config.CELL_SIZE_CM) for (px,py) in path],
                x_cm, y_cm, lookahead_cm)
            desired_heading = math.degrees(math.atan2(look_pt[1]-y_cm, look_pt[0]-x_cm))

            # PID steer toward lookahead
            error = ((desired_heading - hardware.get_heading() + 540) % 360) - 180
            derivative = error
            corr = (config.GYRO_KP*error + config.GYRO_KD*derivative)
            corr = max(min(corr, config.MAX_CORRECTION), -config.MAX_CORRECTION)

            l_spd = max(min(speed_pct - corr + config.LEFT_BIAS, 100), -100)
            r_spd = max(min(speed_pct + corr, 100), -100)
            hardware.tank.on(SpeedPercent(l_spd), SpeedPercent(r_spd))
            time.sleep(0.01)

            # Odometry update if no fresh vision
            if not (robot_pose["x"] is not None and time.time() - robot_pose["timestamp"] < 0.5):
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
        if buf.get('distance_buffer', 0) > 0:
            drive_distance(buf['distance_buffer'])
            buf['distance_buffer'] = 0
        perform_turn(float(cmd['turn']))

    if 'distance' in cmd:
        buf['distance_buffer'] = buf.get('distance_buffer', 0) + float(cmd['distance'])

    if cmd.get('deliver'):
        print("DELIVER")
        _reverse_aux()

    if 'goto' in cmd:
        # flush any buffered encoder distance first
        if buf.get('distance_buffer', 0) > 0:
            drive_distance(buf['distance_buffer'])
            buf['distance_buffer'] = 0
        x, y = cmd['goto']
        drive_to_point(x, y)
        
    if 'face' in cmd:
        if buf.get('distance_buffer', 0) > 0:
            drive_distance(buf['distance_buffer'])
            buf['distance_buffer'] = 0
        rotate_to_heading(float(cmd['face']))








# NEW TO POINT DRIVE

def drive_to_point(target_x_cm, target_y_cm,
                   speed_pct=None, dist_thresh_cm=2.0):
    """Continuously drive toward (x,y) until vision reports we’re within dist_thresh_cm."""
    if speed_pct is None:
        speed_pct = config.DRIVE_SPEED_PCT

    _start_aux()
    try:
        while True:
            # get freshest pose: vision if <0.5s old, else odometry/gyro
            if (motion.robot_pose["x"] is not None
                and time.time() - motion.robot_pose["timestamp"] < 0.5):
                cur_x = motion.robot_pose["x"]
                cur_y = motion.robot_pose["y"]
            else:
                # fallback: you could track with odometry, but here just break
                print("No fresh vision – stopping")
                break

            dx = target_x_cm - cur_x
            dy = target_y_cm - cur_y
            distance = math.hypot(dx, dy)
            if distance <= dist_thresh_cm:
                print("Arrived within {}cm of target".format(dist_thresh_cm))
                break

            # compute desired heading toward target
            desired = math.degrees(math.atan2(dy, dx)) % 360.0
            # heading error via gyro fusion
            current = hardware.get_heading()
            error = ((desired - current + 540) % 360) - 180
            # PID from your config (only P + D here)
            derivative = error  # since last_error resets each loop
            corr = max(min(config.GYRO_KP*error + config.GYRO_KD*derivative,
                           config.MAX_CORRECTION),
                       -config.MAX_CORRECTION)

            l_spd = max(min(speed_pct - corr + config.LEFT_BIAS, 100), -100)
            r_spd = max(min(speed_pct + corr, 100), -100)
            hardware.tank.on(SpeedPercent(l_spd), SpeedPercent(r_spd))
            time.sleep(0.01)

    finally:
        hardware.tank.off()
        _stop_aux()


def rotate_to_heading(target_theta_deg, angle_thresh=1.0):
    """Spin in place until vision+gyro say θ is within angle_thresh."""
    _start_aux()
    try:
        while True:
            # fused current heading
            if (motion.robot_pose["theta"] is not None
                and time.time() - motion.robot_pose["timestamp"] < 0.5):
                current = motion.robot_pose["theta"]
            else:
                current = hardware.get_heading()

            error = ((target_theta_deg - current + 540) % 360) - 180
            if abs(error) <= angle_thresh:
                print("Heading within ".format(angle_thresh))
                break

            # simple P controller on turn speed
            power = max(min(abs(error)*0.5, config.TURN_SPEED_PCT), 5)
            if error > 0:
                hardware.tank.on(-power, power)
            else:
                hardware.tank.on(power, -power)
            time.sleep(0.01)

    finally:
        hardware.tank.off()
        _stop_aux()