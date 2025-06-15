import time
import random
import math
import matplotlib.pyplot as plt
from robot_client import robot_comm
from robot_client.navigation import planner

# Step 1: Connect to mock server
robot_comm.init_robot_connection("127.0.0.1", 12345)

# Step 2: Simulate robot starting pose
robot_x_cm = 20.0
robot_y_cm = 30.0
robot_theta = 0.0
robot_comm.send_pose(robot_x_cm, robot_y_cm, robot_theta)

# Step 3: Initial balls
fake_balls = [(60.0, 30.0), (60.0, 70.0), (30.0, 80.0)]

# Step 4: Plan route
planner.robot_position_cm = (robot_x_cm, robot_y_cm)
route_cm, grid_path = planner.compute_best_route(fake_balls, goal_name="A")
robot_comm.send_path(grid_path, heading=robot_theta)

# For plotting
pose_history = [(robot_x_cm, robot_y_cm, robot_theta)]

# Step 5: Simulate full motion along the route
x, y = robot_x_cm, robot_y_cm
target_index = 1
steps_since_last_progress = 0
last_distance = None

max_total_steps = 1000
step_count = 0

while target_index < len(route_cm) and step_count < max_total_steps:
    step_count += 1
    target_x, target_y = route_cm[target_index]
    dx = target_x - x
    dy = target_y - y
    distance_to_target = math.hypot(dx, dy)

    # Debug output
    print(f"Target {target_index}: (x={target_x:.1f}, y={target_y:.1f}), distance={distance_to_target:.2f}")

    if distance_to_target < 4.0:
        print(f"✅ Reached target {target_index} at distance {distance_to_target:.2f} cm")
        target_index += 1
        steps_since_last_progress = 0
        continue

    forward_theta = math.degrees(math.atan2(dy, dx)) % 360
    theta = (forward_theta + random.uniform(-5.0, 5.0)) % 360

    speed_cm = 1.5
    x += speed_cm * math.cos(math.radians(forward_theta))
    y += speed_cm * math.sin(math.radians(forward_theta))

    lateral_offset = random.uniform(-1.0, 1.0)
    x += lateral_offset * math.cos(math.radians(forward_theta + 90))
    y += lateral_offset * math.sin(math.radians(forward_theta + 90))

    if step_count % 10 == 0:  # Send every 10th step (≈ every 0.18s if sleep is 0.03s)
        robot_comm.send_pose(x, y, theta)
    pose_history.append((x, y, theta))
    time.sleep(0.03)

    if last_distance is not None and abs(distance_to_target - last_distance) < 0.2:
        steps_since_last_progress += 1
    else:
        steps_since_last_progress = 0
    last_distance = distance_to_target

    if steps_since_last_progress > 40:
        print("⚠️ Stuck near target, skipping to next point")
        target_index += 1
        steps_since_last_progress = 0

    if step_count == max_total_steps // 2:
        new_ball = (x + 10, y + 10)
        fake_balls.append(new_ball)
        planner.robot_position_cm = (x, y)
        route_cm, grid_path = planner.compute_best_route(fake_balls, goal_name="A")
        robot_comm.send_path(grid_path, heading=theta)
        target_index = 1
        print("🔄 Replanned path with new ball")

robot_comm.close_robot_connection()

# Plot
plt.figure(figsize=(10, 7))
route_x = [pt[0] for pt in route_cm]
route_y = [pt[1] for pt in route_cm]
plt.plot(route_x, route_y, 'g--o', label='Planned Path', linewidth=2)
plt.plot([p[0] for p in pose_history], [p[1] for p in pose_history], 'r-', label='Simulated Pose', linewidth=1)
plt.scatter(*zip(*fake_balls), c='blue', label='Balls', marker='x', s=100)

plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.title("Simulated Robot Path with Drift and Recovery")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()