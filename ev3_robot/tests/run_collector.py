#!/usr/bin/env python3
import sys
import os
import time

import hardware

if __name__ == '__main__':
    print("Turning aux motor on...")
    hardware.aux_motor.on(20)
    time.sleep(10)
    print("Turning aux motor off.")
    hardware.aux_motor.off()