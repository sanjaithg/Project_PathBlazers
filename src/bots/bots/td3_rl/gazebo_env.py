import math
import random

import numpy as np
import rclpy
from ros_gz_interfaces.srv import SetEntityPose
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Empty
from ros_gz_interfaces.msg import Entity
from visualization_msgs.msg import Marker, MarkerArray

GOAL_REACHED_DIST = 0.4 
COLLISION_DIST = 0.22 
TIME_DELTA = 0.1


def check_pos(x, y):
    obstacles = [
        (-3.8, -6.2, 6.2, 3.8), (-1.3, -2.7, 4.7, -0.2), (-0.3, -4.2, 2.7, 1.3),
        (-0.8, -4.2, -2.3, -4.2), (-1.3, -3.7, -0.8, -2.7), (4.2, 0.8, -1.8, -3.2),
        (4, 2.5, 0.7, -3.2), (6.2, 3.8, -3.3, -4.2), (4.2, 1.3, 3.7, 1.5), (-3.0, -7.2, 0.5, -1.5)
    ]
    if any(x1 > x > x2 and y1 > y > y2 for x1, x2, y1, y2 in obstacles) or not (-4.5 <= x <= 4.5 and -4.5 <= y <= 4.5):
        return False
    return True


class GazeboEnv(Node):
    def __init__(self, environment_dim):
        super().__init__('gazebo_env')
        self.environment_dim = environment_dim
        self.odom_x = 0.0
        self.odom_y = 0.0

        self.goal_x = 1.0
        self.goal_y = 0.0
        
        self.goal_is_fixed = False 
        self.last_distance = 0.0
        self.deviation_counter = 0 

        self.max_distance = 3.5

        self.upper, self.lower = 5.0, -5.0
        self.scan_data = np.ones(self.environment_dim) * self.max_distance
        self.full_scan = np.ones(360) * self.max_distance  # Store FULL lidar for collision
        self.last_odom = None

        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.goal_point_publisher = self.create_publisher(MarkerArray, "goal_point", 3)

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        # Use EKF filtered odometry (fuses wheel odom + IMU for better yaw)
        self.odom_sub = self.create_subscription(Odometry, "/odometry/filtered", self.odom_callback, 10)

        self.set_entity_state = self.create_client(SetEntityPose, "/world/default/set_entity_state")
        self.unpause = self.create_client(Empty, "/world/default/unpause_physics")
        self.pause = self.create_client(Empty, "/world/default/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/world/default/reset")

    def scan_callback(self, scan):
        # Store FULL scan data for collision detection (all 360 or more points)
        ranges = np.nan_to_num(scan.ranges, nan=self.max_distance, posinf=self.max_distance)
        self.full_scan = np.clip(ranges, 1e-6, self.max_distance)
        
        # Sample for state (neural network input)
        mod = max(1, len(scan.ranges) // self.environment_dim)
        self.scan_data = [
            min(self.max_distance, ranges[i]) 
            for i in range(0, len(scan.ranges), mod)
        ]
        self.scan_data = np.maximum(np.array(self.scan_data[:self.environment_dim]), 1e-6)

    def odom_callback(self, msg):
        self.last_odom = msg
    
    def update_goal_status(self, was_goal_reached, is_timeout=False):
        if was_goal_reached:
            self.get_logger().info('Goal reached! New random goal.')
            self.change_goal()
            self.goal_is_fixed = False
        elif is_timeout:
            self.get_logger().info('Timeout. Skipping to new goal.')
            self.change_goal()
            self.goal_is_fixed = False
        else:
            self.get_logger().info('Collision. Keeping goal (Persistence).')
            self.goal_is_fixed = True

    def step(self, action):
        action = [float(a) for a in action]
        if len(action) != 3:
            raise ValueError(f"Action vector must be size 3 for Mecanum, got {len(action)}")

        old_distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        
        vel_cmd = Twist()
        # NO SPEED SCALING - Full speed!
        vel_cmd.linear.x = float(action[0])
        vel_cmd.linear.y = float(action[1])
        vel_cmd.angular.z = float(action[2])
        
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        self.unpause.call_async(Empty.Request())
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=TIME_DELTA))
        self.pause.call_async(Empty.Request())

        done, collision, min_laser = self.observe_collision(self.full_scan)  # Use FULL scan!
        
        if self.last_odom is None:
             self.odom_x = 0.0
             self.odom_y = 0.0
             angle = 0.0
             self.get_logger().warn('No odom data received yet!')
        else:
             # Parse nav_msgs/Odometry
             self.odom_x = float(self.last_odom.pose.pose.position.x)
             self.odom_y = float(self.last_odom.pose.pose.position.y)
             q = self.last_odom.pose.pose.orientation
             r = R.from_quat([q.x, q.y, q.z, q.w])
             euler = r.as_euler('xyz', degrees=False)
             angle = round(euler[2], 4)
             # DEBUG: Log position every 50 steps
             if hasattr(self, '_step_count'):
                 self._step_count += 1
             else:
                 self._step_count = 0
             if self._step_count % 50 == 0:
                 self.get_logger().info(f'[DEBUG] Odom Pose: x={self.odom_x:.2f}, y={self.odom_y:.2f}, yaw={angle:.2f}')

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        
        target_angle = math.atan2(skew_y, skew_x)
        theta = target_angle - angle

        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

        target = distance < GOAL_REACHED_DIST
        
        # Deviation logic
        distance_change = old_distance - distance
        if distance_change <= 0:
            self.deviation_counter += 1
        else:
            self.deviation_counter = 0
            
        if self.deviation_counter > 200:
            self.get_logger().info("Stuck/deviating. Skipping goal.")
            done = True
            self.goal_is_fixed = False
        else:
            done = False
            
        done = done or target or collision
        
        # ===== ENHANCED STATE SPACE =====
        # Add sector-based obstacle proximity for better local awareness
        scan_array = np.array(self.full_scan)
        
        # Calculate danger levels in different sectors (assuming 360-degree scan)
        scan_len = len(scan_array)
        front_start = int(scan_len * 350 / 360)  # -10 to +10 degrees
        front_end = int(scan_len * 10 / 360)
        front_danger = min(np.concatenate([scan_array[front_start:], scan_array[:front_end]]))
        
        left_start = int(scan_len * 70 / 360)   # 70 to 110 degrees
        left_end = int(scan_len * 110 / 360)
        left_danger = np.min(scan_array[left_start:left_end]) if left_end > left_start else self.max_distance
        
        right_start = int(scan_len * 250 / 360)  # 250 to 290 degrees
        right_end = int(scan_len * 290 / 360)
        right_danger = np.min(scan_array[right_start:right_end]) if right_end > right_start else self.max_distance
        
        # Encode heading as sin/cos for circular continuity
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Enhanced robot state: [distance, sin_heading, cos_heading, vx, vy, omega, 
        #                        front_min, left_min, right_min]
        robot_state = [
            distance, 
            sin_theta, 
            cos_theta, 
            action[0], 
            action[1], 
            action[2],
            front_danger / self.max_distance,  # Normalize to [0, 1]
            left_danger / self.max_distance,
            right_danger / self.max_distance
        ]
        
        state = np.append(self.scan_data, robot_state) 

        reward = self.get_reward(target, collision, action, min_laser, old_distance, distance, theta)
        
        return state, reward, done, target

    def reset(self):
        self.deviation_counter = 0
        self.reset_proxy.call_async(Empty.Request())
        
        angle = np.random.uniform(-np.pi, np.pi)

        x = 0.0
        y = 0.0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)

        self.change_object_position("my_robot", x, y, angle)
        
        self.odom_x = float(x)
        self.odom_y = float(y)

        if not self.goal_is_fixed:
            self.random_box() 

        self.publish_markers([0.0, 0.0, 0.0]) 

        self.unpause.call_async(Empty.Request())
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=TIME_DELTA))
        self.pause.call_async(Empty.Request())

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        
        target_angle = math.atan2(skew_y, skew_x)
        theta = target_angle - angle

        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi

        self.last_distance = distance
        
        # ===== ENHANCED STATE SPACE (matching step function) =====
        scan_array = np.array(self.full_scan)
        scan_len = len(scan_array)
        
        # Calculate danger levels in different sectors
        front_start = int(scan_len * 350 / 360)
        front_end = int(scan_len * 10 / 360)
        front_danger = min(np.concatenate([scan_array[front_start:], scan_array[:front_end]]))
        
        left_start = int(scan_len * 70 / 360)
        left_end = int(scan_len * 110 / 360)
        left_danger = np.min(scan_array[left_start:left_end]) if left_end > left_start else self.max_distance
        
        right_start = int(scan_len * 250 / 360)
        right_end = int(scan_len * 290 / 360)
        right_danger = np.min(scan_array[right_start:right_end]) if right_end > right_start else self.max_distance
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        robot_state = [
            distance, 
            sin_theta, 
            cos_theta, 
            0.0,  # vx starts at 0
            0.0,  # vy starts at 0
            0.0,  # omega starts at 0
            front_danger / self.max_distance,
            left_danger / self.max_distance,
            right_danger / self.max_distance
        ]
        
        state = np.append(self.scan_data, robot_state)
        return state
        
    def change_object_position(self, name, x, y, angle):
        r = R.from_euler('xyz', [0.0, 0.0, angle])
        quat = r.as_quat()

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = 0.0
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])

        request = SetEntityPose.Request()
        entity_msg = Entity()
        entity_msg.name = name

        request.entity = entity_msg
        request.pose = pose

        self.set_entity_state.call_async(request)

    def change_goal(self):
        old_goal_x, old_goal_y = self.goal_x, self.goal_y
        
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False
        attempts = 0
        while not goal_ok and attempts < 100:
            attempts += 1
            # Generate new goal within arena
            self.goal_x = random.uniform(-4.0, 4.0)
            self.goal_y = random.uniform(-4.0, 4.0)
            goal_ok = check_pos(self.goal_x, self.goal_y)
        
        self.get_logger().info(f'Goal changed: ({old_goal_x:.2f}, {old_goal_y:.2f}) -> ({self.goal_x:.2f}, {self.goal_y:.2f})')

    def random_box(self):
        for i in range(4):
            name = "cardboard_box_" + str(i)
            x, y = 0.0, 0.0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            self.change_object_position(name, x, y, 0.0)

    def publish_markers(self, action):
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.pose.position.x = float(self.goal_x)
        marker.pose.position.y = float(self.goal_y)
        marker.pose.position.z = 0.0

        markerArray.markers.append(marker)
        self.goal_point_publisher.publish(markerArray)

    def observe_collision(self, laser_data):
        """Soft collision check: return min laser but do NOT trigger a discrete collision flag.
        Collision avoidance and penalties are handled by the CBF-inspired exponential barrier
        inside the reward function (soft penalty), so we avoid binary termination here."""
        min_laser = float(min(laser_data))
        # No hard collision flag—use soft barrier (exponential) in reward instead.
        return False, False, min_laser

    def get_barrier_reward(self, laser_ranges):
        """
        Calculates a Shape-Aware Reciprocal Barrier for a rectangular robot.
        
        Geometry:
        - Robot Hull: 0.4m x 0.2m (Half-dims: 0.20m, 0.10m)
        - Safe Zone:  0.9m x 0.5m (Half_dims: 0.45m, 0.25m)
          (Derived from user specs: Front threshold 0.45m, Side threshold 0.25m)
        
        The barrier activates when the robot enters the Safe Zone and spikes 
        to -2000 if it breaches the Hull (Physical Limit).
        """
        total_barrier = 0.0
        num_rays = len(laser_ranges)
        
        # Physical limits (Hull)
        L_half = 0.20
        W_half = 0.10
        
        # Activation thresholds (Safe Zone)
        # Front/Back braking distance: 0.20 + 0.25 = 0.45
        # Side corridor cushion: 0.10 + 0.15 = 0.25
        L_safe = 0.45
        W_safe = 0.25
        
        angle_step = 2 * math.pi / num_rays
        
        for i, dist in enumerate(laser_ranges):
            # FIX: ROS LaserScan usually starts at -pi (Back), not 0 (Front)
            # Index 0 = -pi (Back), Index N/2 = 0 (Front)
            angle = -math.pi + i * angle_step
            
            abs_cos = abs(math.cos(angle))
            abs_sin = abs(math.sin(angle))

            # 1. Calculate Physical Limit at this angle (Rectangular Hull)
            # radius_limit = 1 / max(cos/L, sin/W)
            # Avoid division by zero with epsilon
            phys_limit = 1.0 / max(abs_cos / L_half, abs_sin / W_half)
            
            # 2. Calculate Activation Threshold at this angle (Rectangular Safe Zone)
            safe_thresh = 1.0 / max(abs_cos / L_safe, abs_sin / W_safe)
            
            if dist < phys_limit:
                return -2000.0  # Hard collision cap
            elif dist < safe_thresh:
                # Reciprocal penalty
                # As dist approaches phys_limit, (dist - phys_limit) -> 0, penalty -> -inf
                # We add small epsilon to denominator for stability
                penalty = -1.0 / (dist - phys_limit + 1e-4)
                total_barrier += penalty
                
        return total_barrier

    def get_reward(self, target, collision, action, min_laser, old_distance, new_distance, theta):
        """
        CBF-inspired soft barrier reward with normalized weights.

        Components:
        - R_progress (w1=0.40): distance improvement
        - R_barrier (w2=0.35): SHAPE-AWARE reciprocal barrier
        - R_heading (w3=0.05): cosine alignment with goal
        - R_smoothness (w4=0.05): penalize jerky actions
        - R_angular (w5=0.05): penalize excessive spinning
        - R_existence (w6=0.10): small positive time reward to encourage exploration
        Total weights sum to 1.0.
        """
        # Terminal (goal reached) is still rewarded strongly
        if target:
            return 500.0

        # ===== 1. PROGRESS =====
        distance_change = old_distance - new_distance
        R_progress = distance_change * 200.0  # same scale as before

        # ===== 2. SHAPE-AWARE BARRIER =====
        # Use the 20-ray scan_data for the barrier calculation
        # This matches the dimensionality of the user's snippet
        R_barrier = self.get_barrier_reward(self.scan_data)
        
        # Clamp R_barrier to the user-approved safety cap (-2000) for the final weighted sum
        # to prevent one bad ray from destroying the epoch average if not a hard collision
        R_barrier = max(R_barrier, -2000.0)


        # ===== 3. HEADING =====
        R_heading = math.cos(theta)

        # ===== 4. SMOOTHNESS =====
        action_arr = np.array(action)
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.array([0.0, 0.0, 0.0])
        action_change = np.linalg.norm(action_arr - self.prev_action)
        R_smoothness = -action_change
        self.prev_action = action_arr.copy()

        # ===== 5. ANGULAR =====
        R_angular = -abs(action[2])

        # ===== 6. EXISTENCE =====
        R_existence = 0.1

        # Normalize / scale components to reasonable ranges
        # Progress: scale same as before
        R_progress_scaled = R_progress  # already scaled
        
        # Barrier: currently R_barrier can be -2000. 
        # We need to balance this with the 0.35 weight.
        # If R_barrier is -2000, 0.35 * -2000 = -700. This is huge compared to +4.0 progress.
        # This is INTENTIONAL as per user request ("dominate progress reward").
        R_barrier_scaled = R_barrier * 1.0
        
        R_heading_scaled = R_heading * 5.0  # keep similar magnitude influence
        R_smoothness_scaled = R_smoothness * 5.0
        R_angular_scaled = R_angular * 3.0
        R_existence_scaled = R_existence * 1.0

        # Weighted sum (weights sum to 1.0)
        total_reward = (
            0.40 * R_progress_scaled +
            0.35 * R_barrier_scaled +
            0.05 * R_heading_scaled +
            0.05 * R_smoothness_scaled +
            0.05 * R_angular_scaled +
            0.10 * R_existence_scaled
        )

        return total_reward