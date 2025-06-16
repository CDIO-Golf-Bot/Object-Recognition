#!/usr/bin/env python3

import sys
import os

# Add the script directory and its 'python' subdirectory to the system path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'python'))

from server import run_server

if __name__=='__main__':
    run_server()