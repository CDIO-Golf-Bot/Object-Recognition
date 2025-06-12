#!/usr/bin/env python3
"""
Simple client script to send a hardcoded path to the robot server.
Run: python client.py
"""
import socket
import json
import time

# === CONFIGURATION ===
HOST = '192.168.0.100'  # change to your robot's IP
PORT = 12345

# Hardcoded path: square, 2cm grid coordinates
path = [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]
heading = 'E'  # initial heading

cmd = {'path': path, 'heading': heading}

def send_path(cmd, host=HOST, port=PORT):
    try:
        with socket.create_connection((host, port), timeout=5) as sock:
            msg = json.dumps(cmd) + '\n'
            sock.sendall(msg.encode())
            print("Sent path to {}:{}".format(host, port))
    except Exception as e:
        print("Failed to send path:", e)

if __name__ == '__main__':
    print("Client starting...")
    send_path(cmd)
    time.sleep(1)
    print("Done.")
