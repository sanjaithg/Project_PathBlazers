import math
import random
import time

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


# def check_pos(x, y):
#     obstacles = [
#         (-3.8, -6.2, 6.2, 3.8), (-1.3, -2.7, 4.7, -0.2), (-0.3, -4.2, 2.7, 1.3),
#         (-0.8, -4.2, -2.3, -4.2), (-1.3, -3.7, -0.8, -2.7), (4.2, 0.8, -1.8, -3.2),
#         (4, 2.5, 0.7, -3.2), (6.2, 3.8, -3.3, -4.2), (4.2, 1.3, 3.7, 1.5), (-3.0, -7.2, 0.5, -1.5)
#     ]
#     if any(x1 > x > x2 and y1 > y > y2 for x1, x2, y1, y2 in obstacles) or not (-4.5 <= x <= 4.5 and -4.5 <= y <= 4.5):
#         return False
#     return True

def check_pos_simplified(x, y):
    # Simplified bounds check for empty arena (10x10)
    if not (-4.5 <= x <= 4.5 and -4.5 <= y <= 4.5):
        return False
    return True


class GazeboEnv(Node):
    def __init__(self, environment_dim, use_ground_truth=False, pose_topic='/model/my_robot/pose', odom_topic='/odometry/filtered', ground_truth_noise_std=0.0):
        super().__init__('gazebo_env')
        self.environment_dim = environment_dim
        
        # ROS Parameters
        self.declare_parameter('use_ground_truth', use_ground_truth)
        self.declare_parameter('pose_topic', pose_topic)
        self.declare_parameter('odom_topic', odom_topic)
        self.declare_parameter('ground_truth_noise_std', ground_truth_noise_std)

        self.use_ground_truth = self.get_parameter('use_ground_truth').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.ground_truth_noise_std = self.get_parameter('ground_truth_noise_std').value

        if self.use_ground_truth:
            self.get_logger().info(f'Using ground-truth pose ({self.pose_topic}) with noise={self.ground_truth_noise_std}')
        else:
            self.get_logger().info(f'Using EKF odom ({self.odom_topic})')

        self.odom_x = 0.0
        self.odom_y = 0.0

        self.goal_x = 1.0
        self.goal_y = 0.0
        
        self.goal_is_fixed = False 
        self.last_distance = 0.0
        self.deviation_counter = 0 
        
        # Action history for jitter reduction (last 2 actions)
        self.action_history = np.zeros(6) 

        self.max_distance = 3.5

        self.upper, self.lower = 5.0, -5.0
        self.scan_data = np.ones(self.environment_dim) * self.max_distance
        self.full_scan = np.ones(360) * self.max_distance  # Store FULL lidar for collision
        self.last_odom = None
        self.last_pose = None
        self._fallback_logged = False

        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.goal_point_publisher = self.create_publisher(MarkerArray, "goal_point", 3)

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        # Subscription to both sources
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.pose_sub = self.create_subscription(Odometry, self.pose_topic, self.pose_callback, 10)

        self.set_entity_state = self.create_client(SetEntityPose, "/world/default/set_pose")
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
        self.last_odom_time = self.get_clock().now().nanoseconds

    def pose_callback(self, msg):
        # Extract Pose from Odometry message
        self.get_logger().info("CB")
        self.last_pose = msg.pose.pose
        self.last_pose_time = self.get_clock().now().nanoseconds

    def get_current_pose(self):
        """Returns (x, y, yaw) with preference/fallback/noise logic."""
        use_fallback = False
        pose_to_parse = None
        source_name = ""

        if self.use_ground_truth:
            if self.last_pose is not None:
                pose_to_parse = self.last_pose
                source_name = "ground-truth"
            else:
                use_fallback = True
                if not self._fallback_logged:
                    self.get_logger().warn(f'Ground-truth pose ({self.pose_topic}) missing! Falling back to odom.')
                    self._fallback_logged = True
        
        if not self.use_ground_truth or use_fallback:
            if self.last_odom is not None:
                pose_to_parse = self.last_odom.pose.pose
                source_name = "odom"
                if self.use_ground_truth and not use_fallback: # Shouldn't happen usually
                     source_name = "odom (fallback)"
            else:
                if not hasattr(self, '_warning_logged') or not self._warning_logged:
                    self.get_logger().warn('No pose or odom data received yet!')
                    self._warning_logged = True
                return None

        # Parse position
        x = float(pose_to_parse.position.x)
        y = float(pose_to_parse.position.y)

        # Apply noise if ground truth and requested
        if source_name == "ground-truth" and self.ground_truth_noise_std > 0.0:
            x += np.random.normal(0, self.ground_truth_noise_std)
            y += np.random.normal(0, self.ground_truth_noise_std)

        # Parse orientation (yaw)
        q = pose_to_parse.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        euler = r.as_euler('xyz', degrees=False)
        angle = round(euler[2], 4)

        if source_name == "ground-truth" and self.ground_truth_noise_std > 0.0:
            # Noise for orientation (rad)
            angle += np.random.normal(0, self.ground_truth_noise_std)
            # Wrap angle
            angle = (angle + np.pi) % (2 * np.pi) - np.pi

        return x, y, angle
    
    def update_goal_status(self, was_goal_reached, is_timeout=False):
        if was_goal_reached:
            self.get_logger().info('Goal reached! New random goal.')
            self.change_goal()
            self.goal_is_fixed = False
            self.deviation_counter = 0  # Reset deviation counter for new goal
        elif is_timeout:
            self.get_logger().info('Timeout. Skipping to new goal.')
            self.change_goal()
            self.goal_is_fixed = False
            self.deviation_counter = 0  # Reset deviation counter for new goal
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

        # self.unpause.call_async(Empty.Request())
        
        # Wait for valid NEW pose data
        # We expect a position update to happen during this window.
        # Instead of fixed sleep, we wait for a timestamp update.
        
        # 1. Minimum sleep for physics to run
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=TIME_DELTA))
        
        # 2. Wait for callback to fire (max 0.2s extra)
        # Note: If topic is dead, this will timeout and use OLD pose (0 move)
        # This prevents infinite hanging.
        wait_start = self.get_clock().now()
        got_new_pose = False
        while (self.get_clock().now() - wait_start).nanoseconds < 1e9: # 1.0s max wait
             if self.use_ground_truth and hasattr(self, 'last_pose_time'):
                 if (self.get_clock().now().nanoseconds - self.last_pose_time) < 3e8: # New within 0.3s
                     got_new_pose = True
                     break
             elif hasattr(self, 'last_odom_time'):
                 if (self.get_clock().now().nanoseconds - self.last_odom_time) < 3e8:
                     got_new_pose = True
                     break
             time.sleep(0.005) # Yield
        
        if not got_new_pose:
             self.get_logger().warn("[STEP] Pose update TIMEOUT! Physics likely stuck.")

        # self.pause.call_async(Empty.Request())

        min_laser = self.observe_collision(self.full_scan)  # Use FULL scan!
        collision = False
        
        if min_laser < COLLISION_DIST:
            self.get_logger().info("Collision detected!")
            collision = True

        done = False
        
        pose_info = self.get_current_pose()
        if pose_info is None:
             self.odom_x = 0.0
             self.odom_y = 0.0
             angle = 0.0
        else:
             self.odom_x, self.odom_y, angle = pose_info
             # DEBUG: Log position every 50 steps
             if hasattr(self, '_step_count'):
                 self._step_count += 1
             else:
                 self._step_count = 0
             if self._step_count % 50 == 0:
                 self.get_logger().info(f'[DEBUG] {"GT" if self.use_ground_truth else "Odom"} Pose: x={self.odom_x:.2f}, y={self.odom_y:.2f}, yaw={angle:.2f}')

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
        
        # Calculate danger levels in different sectors with wraparound safety
        scan_len = len(scan_array) if len(scan_array) > 0 else 1
        
        # Front sector: -10 to +10 degrees (wraparound)
        front_start = int(scan_len * 350 / 360)
        front_end = int(scan_len * 10 / 360)
        front_indices = list(range(front_start, scan_len)) + list(range(0, front_end))
        front_danger = float(np.min([scan_array[i] for i in front_indices])) if front_indices else self.max_distance
        
        # Left sector: 70 to 110 degrees
        left_start = int(scan_len * 70 / 360)
        left_end = int(scan_len * 110 / 360)
        left_danger = float(np.min(scan_array[left_start:left_end])) if left_end > left_start else self.max_distance
        
        # Right sector: 250 to 290 degrees
        right_start = int(scan_len * 250 / 360)
        right_end = int(scan_len * 290 / 360)
        right_danger = float(np.min(scan_array[right_start:right_end])) if right_end > right_start else self.max_distance
        
        # Encode goal in local coordinates instead of distance/heading
        # target_x_local = distance * cos_theta
        # target_y_local = distance * sin_theta
        target_x_local = distance * np.cos(theta)
        target_y_local = distance * np.sin(theta)
        
        # Enhanced robot state: [target_x_local, target_y_local, vx, vy, omega, 
        #                        front_danger, left_min, right_min, action_history (6)]
        robot_state = [
            target_x_local,
            target_y_local,
            action[0], 
            action[1], 
            action[2],
            front_danger / self.max_distance,  # Normalize to [0, 1]
            left_danger / self.max_distance,
            right_danger / self.max_distance
        ]
        
        # Add action history to state
        robot_state.extend(self.action_history.tolist())
        
        # Update action history for NEXT step
        self.action_history = np.append(action, self.action_history[:3])
        
        state = np.append(self.scan_data, robot_state) 

        reward, step_info = self.get_reward(target, collision, action, min_laser, old_distance, distance, theta)
        
        # STUCK DETECTION
        is_stuck = self.check_stuck(self.odom_x, self.odom_y)
        if is_stuck and not done:
            self.get_logger().warn("Robot is STUCK! Ending Episode to reset.")
            done = True
            reward = -500.0  # Penalty for getting stuck
            
        # Add pose and goal info to step_info for CSV logging
        step_info.update({
            'pose_x': self.odom_x,
            'pose_y': self.odom_y,
            'pose_yaw': angle,
            'goal_x': self.goal_x,
            'goal_y': self.goal_y,
            'min_laser': min_laser,
            'distance_to_goal': distance,
            'cmd_vel_x': action[0],
            'cmd_vel_y': action[1],
            'cmd_vel_z': action[2]
        })
        
        return state, reward, done, target, step_info

    def reset(self):
        self.deviation_counter = 0
        # Initialize reward tracking variables to prevent cross-episode contamination
        self.prev_theta = 0.0
        self.prev_action = np.array([0.0, 0.0, 0.0])
        self.reset_proxy.call_async(Empty.Request())
        
        angle = np.random.uniform(-np.pi, np.pi)

        x = 0.0
        y = 0.0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos_simplified(x, y)

        commanded_angle = angle  # Save commanded angle as fallback
        self.change_object_position("my_robot", x, y, angle)
        
        # self.unpause.call_async(Empty.Request())
        
        # Wait for a valid pose update after teleporting
        wait_start = time.time()
        pose_info = None
        while pose_info is None and (time.time() - wait_start) < 2.0:
            # DO NOT SPIN HERE! Executor handles it.
            time.sleep(0.1)
            pose_info = self.get_current_pose()
            
        if pose_info is not None:
            self.odom_x, self.odom_y, angle = pose_info
        else:
            self.get_logger().warn("Reset: Failed to get pose after 2s, using commanded position.")
            self.odom_x = float(x)
            self.odom_y = float(y)
            angle = commanded_angle  # Use saved commanded angle as fallback
        
        # self.get_clock().sleep_for(rclpy.duration.Duration(seconds=TIME_DELTA))
        
        # self.pause.call_async(Empty.Request())

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
        
        # Stuck Detection Init
        self.stuck_buffer = []  # History of positions
        self.stuck_threshold = 0.05  # m
        self.stuck_window = 50       # steps ~ 5 seconds
        
        # ===== ENHANCED STATE SPACE (matching step function) =====
        scan_array = np.array(self.full_scan)
        scan_len = len(scan_array) if len(scan_array) > 0 else 1
        
        # Calculate danger levels in different sectors with wraparound safety
        
        # Front sector: -10 to +10 degrees (wraparound)
        front_start = int(scan_len * 350 / 360)
        front_end = int(scan_len * 10 / 360)
        front_indices = list(range(front_start, scan_len)) + list(range(0, front_end))
        front_danger = float(np.min([scan_array[i] for i in front_indices])) if front_indices else self.max_distance
        
        # Left sector: 70 to 110 degrees
        left_start = int(scan_len * 70 / 360)
        left_end = int(scan_len * 110 / 360)
        left_danger = float(np.min(scan_array[left_start:left_end])) if left_end > left_start else self.max_distance
        
        # Right sector: 250 to 290 degrees
        right_start = int(scan_len * 250 / 360)
        right_end = int(scan_len * 290 / 360)
        right_danger = float(np.min(scan_array[right_start:right_end])) if right_end > right_start else self.max_distance
        
        # Initialize action history on reset
        self.action_history = np.zeros(6)
        
        # Encode goal in local coordinates
        target_x_local = distance * np.cos(theta)
        target_y_local = distance * np.sin(theta)
        
        robot_state = [
            target_x_local,
            target_y_local,
            0.0,  # vx starts at 0
            0.0,  # vy starts at 0
            0.0,  # omega starts at 0
            front_danger / self.max_distance,
            left_danger / self.max_distance,
            right_danger / self.max_distance
        ]
        # Initialize with zeros for action history in first state
        robot_state.extend(self.action_history.tolist())
        
        state = np.append(self.scan_data, robot_state)
        return state
        
    def change_object_position(self, name, x, y, angle):
        r = R.from_euler('xyz', [0.0, 0.0, angle])
        quat = r.as_quat()

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = 0.05  # Slight lift to avoid ground collision
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])

        request = SetEntityPose.Request()
        entity_msg = Entity()
        entity_msg.name = name

        request.entity = entity_msg
        request.pose = pose

        # Check if service is available
        if not self.set_entity_state.service_is_ready():
            self.get_logger().warn(f'[TELEPORT] Service /world/default/set_pose NOT ready!')
            self.set_entity_state.wait_for_service(timeout_sec=2.0)
        
        self.get_logger().info(f'[TELEPORT] Moving {name} to ({x:.2f}, {y:.2f}, yaw={angle:.2f})')
        future = self.set_entity_state.call_async(request)
        
        # Wait for service response
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if future.done():
            result = future.result()
            if result is not None:
                self.get_logger().info(f'[TELEPORT] Success: {result.success}')
            else:
                self.get_logger().warn('[TELEPORT] Service returned None!')
        else:
            self.get_logger().warn('[TELEPORT] Service call timed out!')
        
        # Give Gazebo time to settle after teleport
        time.sleep(0.2)

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
            # Generate new goal within arena (conservative bounds for safety)
            self.goal_x = random.uniform(-3.5, 3.5)
            self.goal_y = random.uniform(-3.5, 3.5)
            goal_ok = check_pos_simplified(self.goal_x, self.goal_y)
        
        self.get_logger().info(f'[GOAL] Changed: ({old_goal_x:.2f}, {old_goal_y:.2f}) -> ({self.goal_x:.2f}, {self.goal_y:.2f})')

    def random_box(self):
        for i in range(4):
            name = "cardboard_box_" + str(i)
            x, y = 0.0, 0.0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos_simplified(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            self.change_object_position(name, x, y, 0.0)

    def publish_markers(self, action):
        markerArray = MarkerArray()
        marker = Marker()
        # Use 'map' frame for stable visualization (not 'odom' which drifts)
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0  # Fixed ID for proper marker updates
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.5  # Larger for visibility
        marker.scale.y = 0.5
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.lifetime.sec = 0  # Persistent marker
        
        marker.pose.position.x = float(self.goal_x)
        marker.pose.position.y = float(self.goal_y)
        marker.pose.position.z = 0.05  # Slightly above ground

        markerArray.markers.append(marker)
        self.goal_point_publisher.publish(markerArray)

    def observe_collision(self, laser_data):
        """Extract minimum laser distance for reward calculation.
        Collision avoidance is handled by the CBF-inspired barrier in get_reward().
        """
        return float(min(laser_data))

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
                # Reciprocal penalty with immediate clamping
                # As dist approaches phys_limit, (dist - phys_limit) -> 0, penalty -> -inf
                # We clamp denominator to prevent overflow
                denom = max(dist - phys_limit, 1e-3)  # Enforce minimum denominator
                penalty = -1.0 / denom
                penalty = max(penalty, -2000.0)  # Clamp immediately
                total_barrier += penalty
                
        return total_barrier

    def check_stuck(self, current_x, current_y):
        self.stuck_buffer.append((current_x, current_y))
        if len(self.stuck_buffer) > self.stuck_window:
            self.stuck_buffer.pop(0)
            
            # Check displacement over the window
            start_x, start_y = self.stuck_buffer[0]
            dist_moved = math.hypot(current_x - start_x, current_y - start_y)
            
            if dist_moved < self.stuck_threshold:
                return True
        return False

    def get_reward(self, target, collision, action, min_laser, old_distance, new_distance, theta):
        """
        CBF-inspired soft barrier reward with modern components:
        - Potential-based progress reward (PBRS)
        - Velocity toward goal reward
        - Adaptive barrier scaling with floor and weight compensation
        - Enhanced heading reward (alignment + improvement)
        - Smoothness, angular, existence penalties
        """
        # Terminal reward for reaching the goal
        if target:
            # Return tuple to match unpack signature in step()
            return 2500.0, {
                'total': 2500.0,
                'progress': 0.0,
                'velocity': 0.0,
                'barrier': 0.0,
                'heading': 0.0,
                'smoothness': 0.0,
                'angular': 0.0,
                'existence': 0.0
            }

        # ----- 1. POTENTIAL-BASED PROGRESS REWARD -----
        phi_old = -old_distance  # potential = -distance
        phi_new = -new_distance
        
        
        # Reward V3: Filtered & Focused
        R_progress_pbrs = (phi_new - phi_old) * 1000.0

        # ----- 2. BARRIER (SAFETY) -----
        R_barrier = self.get_barrier_reward(self.full_scan)
        R_barrier = max(R_barrier, -2000.0)  # safety cap
        
        # ----- 3. WEIGHTED SUM -----
        # Only Progress and Barrier matter. Goal is handled in return statement above.
        total_reward = (
            1.0 * R_progress_pbrs +
            1.0 * R_barrier
        )
        
        # Update prev state vars just in case we need them later (though removed from reward)
        if not hasattr(self, 'prev_theta'): self.prev_theta = theta
        self.prev_theta = theta
        if not hasattr(self, 'prev_action'): self.prev_action = np.zeros(3)
        self.prev_action = np.array(action)

        info_dict = {
            'total': total_reward,
            'progress': R_progress_pbrs,
            'velocity': 0.0,
            'barrier': R_barrier,
            'heading': 0.0,
            'smoothness': 0.0,
            'angular': 0.0,
            'existence': 0.0
        }
        
        return total_reward, info_dict