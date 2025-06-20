#!/usr/bin/env python3

import sys
import os
import time

# Add the script directory and its 'python' subdirectory to the system path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'python'))

import motion
import hardware

if __name__=='__main__':
    print("Rotating to 90 degrees...")
    motion.rotate_to_heading(90)
    print("Done. Current heading:", hardware.get_heading())
    time.sleep(0.3)
    motion.rotate_to_heading(90)
    print("Done. Current heading:", hardware.get_heading())
    time.sleep(0.3)
    motion.rotate_to_heading(180)
    print("Done. Current heading:", hardware.get_heading())
    time.sleep(2)