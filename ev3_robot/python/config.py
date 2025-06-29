import math

# ————————————————————————
# WHEELS & FIELD
# ————————————————————————
WHEEL_DIAM_CM   = 4.13
WHEEL_CIRC_CM   = WHEEL_DIAM_CM * math.pi
CELL_SIZE_CM    = 2.0

# ————————————————————————
# GYRO PID
# ————————————————————————
GYRO_KP         = 1.2
GYRO_KI         = 0.0003
GYRO_KD         = 2.0
MAX_CORRECTION  = 15.0
ANGLE_TOLERANCE = 1.0
LEFT_BIAS       = 0.0
RIGHT_BIAS      = 0.0
ANGLE_OVERSHOOT = 5.0      # turns if overshoot, for when missing target
DISTANCE_TRESHHOLD = 8.0    # when we say we hit the target
# ————————————————————————
# SPEEDS
# ————————————————————————
TURN_SPEED_PCT  = 15
DRIVE_SPEED_PCT = 35
FEED_FORWARD    = 2.0
APPROACH_DISTANCE  = 0.1       # distance for when the robot slows down almost turn off lower than 5
# ————————————————————————
# AUX MOTOR
# ————————————————————————
AUX_FORWARD_PCT = 25
AUX_REVERSE_PCT = -30
AUX_REVERSE_SEC = 8.0

# Number of encoder ticks per full wheel revolution
TICKS_PER_REV = 360  # or whatever your hardware uses

# Maximum linear speed of the robot in cm/s at 100% motor power
default_MAX_LINEAR_SPEED_CM_S = 20.0  # calibrate me!

TICKS_PER_REV = 360        # your motor’s encoder ticks per wheel rev
MAX_LINEAR_SPEED_CM_S = 20 # calibrate: robot speed in cm/s at 100%

MAX_ARUCO_AGE   = 0.6
LOG_INTERVAL = 1.0