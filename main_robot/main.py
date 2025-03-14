#!/usr/bin/env pybricks-micropython

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile


# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.


# Create your objects here.
ev3 = EV3Brick()
left_motor = Motor(Port.B, Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.C, Direction.COUNTERCLOCKWISE)
front_motor = Motor(Port.D, Direction.COUNTERCLOCKWISE)

# Write your program here.
ev3.speaker.beep()
robot = DriveBase(left_motor, right_motor, wheel_diameter=54, axle_track = 200)

# Move the front motor forward for 2 seconds
front_motor.run_time(-200, 20000)  # speed: 500 (degrees per second), time: 2000ms (2 seconds)

# Alternatively, reverse the front motor for 2 seconds
front_motor.run_time(150, 4000)  # speed: -500 (negative speed for reverse), time: 2000ms (2 seconds)

# Stop the motor after movement
front_motor.stop()

#robot.straight(200)
#robot.straight(-200)

#robot.turn(180)