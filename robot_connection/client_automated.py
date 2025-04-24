import socket
import time

def get_movement_commands(current, target, heading):
    """Determines the movement commands needed to go from current to target coordinates.
       Returns a tuple: (list_of_commands, desired_heading)"""
    x1, y1 = current
    x2, y2 = target

    # Determine desired heading based on coordinate differences
    if x2 > x1:
        desired_heading = 'E'
    elif x2 < x1:
        desired_heading = 'W'
    elif y2 > y1:
        desired_heading = 'N'
    else:
        desired_heading = 'S'
    
    turn_map = {
        ('N', 'E'): 'right', ('N', 'W'): 'left', ('N', 'S'): 'turn around',
        ('E', 'S'): 'right', ('E', 'N'): 'left', ('E', 'W'): 'turn around',
        ('S', 'W'): 'right', ('S', 'E'): 'left', ('S', 'N'): 'turn around',
        ('W', 'N'): 'right', ('W', 'S'): 'left', ('W', 'E'): 'turn around',
    }
    
    commands = []
    # If the current heading does not match the desired, add the turning command
    if heading != desired_heading:
        turn_cmd = turn_map.get((heading, desired_heading), 'turn error')
        commands.append(turn_cmd)
    
    # After turning, move forward
    commands.append('forward')
    return commands, desired_heading

# Client Code
def send_path(host, port, path, heading):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
        print("Connected to EV3")
        current_pos = path[0]
        
        for target_pos in path[1:]:
            commands, desired_heading = get_movement_commands(current_pos, target_pos, heading)
            
            for cmd in commands:
                print(f"Sending command: {cmd}")
                client_socket.send(cmd.encode())  # Send command
                time.sleep(1)  # Small delay to ensure robot processes one command at a time
            
            # Update current position and heading
            current_pos = target_pos
            heading = desired_heading
            
    except ConnectionRefusedError:
        print("Could not connect to EV3. Is the server running?")
    finally:
        client_socket.close()
        print("Disconnected from EV3.")

# Example path hardcoded
heading = 'S' # South
path = [(1, 1), (2, 1), (3, 1), (3, 0), (3, -1), (2, -1)]
send_path("10.225.58.57", 12345, path, heading)
