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
GYRO_KP         = 3.0
GYRO_KI         = 0.0003
GYRO_KD         = 6.0
MAX_CORRECTION  = 12.0
ANGLE_TOLERANCE = 5.0
LEFT_BIAS       = 0.0
RIGHT_BIAS      = 0.0

# ————————————————————————
# SPEEDS
# ————————————————————————
TURN_SPEED_PCT  = 30
DRIVE_SPEED_PCT = 35

# ————————————————————————
# AUX MOTOR
# ————————————————————————
AUX_FORWARD_PCT = 25
AUX_REVERSE_PCT = -25
AUX_REVERSE_SEC = 5.0
