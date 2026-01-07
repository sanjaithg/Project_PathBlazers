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


def check_pos(x, y):
    obstacles = [
        (-3.8, -6.2, 6.2, 3.8), (-1.3, -2.7, 4.7, -0.2), (-0.3, -4.2, 2.7, 1.3),
        (-0.8, -4.2, -2.3, -4.2), (-1.3, -3.7, -0.8, -2.7), (4.2, 0.8, -1.8, -3.2),
        (4, 2.5, 0.7, -3.2), (6.2, 3.8, -3.3, -4.2), (4.2, 1.3, 3.7, 1.5), (-3.0, -7.2, 0.5, -1.5)
    ]
    # Strict boundary check: Robot must stay within the main arena
    # Arena is roughly 10x10, so +/- 4.5 is a safe internal margin.
    x_min, x_max = -4.5, 4.5
    y_min, y_max = -4.5, 4.5
    
    if x < x_min or x > x_max or y < y_min or y > y_max:
        return False

    # Check against internal obstacles
    if any(x1 > x > x2 and y1 > y > y2 for x1, x2, y1, y2 in obstacles):
        return False
        
    return True


class GazeboEnv(Node):
    def __init__(self, environment_dim, use_ground_truth=False, pose_topic='/model/my_robot/odometry_with_covariance', odom_topic='/odometry/filtered', ground_truth_noise_std=0.0):
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

        self.max_distance = 3.5

        self.upper, self.lower = 5.0, -5.0
        self.scan_data = np.ones(self.environment_dim) * self.max_distance
        self.full_scan = np.ones(360) * self.max_distance  # Store FULL lidar for collision
        self.last_odom = None
        self.last_pose = None
        self._fallback_logged = False
        self._startup_time = time.time()
        self._pose_grace_period = 3.0  # seconds to wait before warning about missing pose

        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.goal_point_publisher = self.create_publisher(MarkerArray, "goal_point", 3)

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        # Subscription to both sources (both are now Odometry messages)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.gt_odom_sub = self.create_subscription(Odometry, self.pose_topic, self.gt_odom_callback, 10)

        self.set_entity_state = self.create_client(SetEntityPose, "/world/default/set_entity_state")
        self.unpause = self.create_client(Empty, "/world/default/unpause_physics")
        self.pause = self.create_client(Empty, "/world/default/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/world/default/reset")

        # Wait for initial pose data if using ground truth
        if self.use_ground_truth:
            self.get_logger().info(f'Waiting for ground-truth pose on {self.pose_topic}...')
            
            # First check if topic exists and has publishers
            max_topic_wait = 30.0  # Wait up to 30s for topic/publisher
            topic_wait_start = time.time()
            topic_found = False
            
            while not topic_found and (time.time() - topic_wait_start) < max_topic_wait:
                topic_names_and_types = self.get_topic_names_and_types()
                topic_exists = any(topic_name == self.pose_topic for topic_name, _ in topic_names_and_types)
                
                if topic_exists:
                    # Topic exists, check for publishers
                    pub_count = self.count_publishers(self.pose_topic)
                    if pub_count > 0:
                        self.get_logger().info(f'Topic {self.pose_topic} found with {pub_count} publisher(s)!')
                        topic_found = True
                        break
                    else:
                        self.get_logger().info(f'Topic exists but no publishers yet, waiting...')
                else:
                    self.get_logger().info(f'Topic {self.pose_topic} not found yet, waiting for bridge/spawn...')
                
                time.sleep(1.0)
            
            if not topic_found:
                self.get_logger().error(f'Topic {self.pose_topic} not available after {max_topic_wait}s! Is Gazebo/bridge running?')
            
            # Now wait for actual pose data
            wait_start = time.time()
            while self.last_pose is None and (time.time() - wait_start) < 10.0:
                rclpy.spin_once(self, timeout_sec=0.1)
            
            if self.last_pose is not None:
                self.get_logger().info('Ground-truth pose received!')
            else:
                self.get_logger().warn(f'No ground-truth pose data after waiting. Will use fallback if available.')

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

    def gt_odom_callback(self, msg):
        """Callback for ground truth odometry from Gazebo."""
        # Store the pose part from the Odometry message
        self.last_pose = msg.pose.pose
        # DEBUG: Uncomment to verify callback is firing
        # self.get_logger().info(f'[GT_CALLBACK] x={msg.pose.pose.position.x:.2f}, y={msg.pose.pose.position.y:.2f}')

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
                # Check if we're still in grace period
                elapsed = time.time() - self._startup_time
                if elapsed < self._pose_grace_period:
                    # Still in grace period, don't warn yet
                    use_fallback = True
                else:
                    use_fallback = True
                    if not self._fallback_logged:
                        self.get_logger().warn(f'Ground-truth pose ({self.pose_topic}) missing after {self._pose_grace_period}s! Falling back to odom.')
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
            self.deviation_counter = 0
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
        
        # Wait for physics to run and for new sensor data to arrive
        self.get_clock().sleep_for(rclpy.duration.Duration(seconds=TIME_DELTA))
        
        # Process pending callbacks to get fresh LIDAR and pose data
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.01)
        
        self.pause.call_async(Empty.Request())

        min_laser = self.observe_collision(self.full_scan)
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
            
        done = done or target

        
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
        left_indices = range(left_start, left_end) if left_end > left_start else []
        if left_indices:
            left_danger = np.min(scan_array[left_start:left_end])
        else:
            left_danger = self.max_distance
        
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

        # Check for collision based on COLLISION_DIST
        collision = min_laser < COLLISION_DIST
        if collision:
            done = True
        
        reward, reward_info = self.get_reward(target, collision, action, min_laser, old_distance, distance, theta)
        
        # Add robot state info for logging
        reward_info['cmd_vel_x'] = action[0]
        reward_info['cmd_vel_y'] = action[1]
        reward_info['cmd_vel_z'] = action[2]
        reward_info['pose_x'] = self.odom_x
        reward_info['pose_y'] = self.odom_y
        reward_info['pose_yaw'] = angle
        reward_info['goal_x'] = self.goal_x
        reward_info['goal_y'] = self.goal_y
        
        return state, reward, done, target, reward_info

    def reset(self):
        self.deviation_counter = 0
        self.reset_proxy.call_async(Empty.Request())
        
        commanded_angle = np.random.uniform(-np.pi, np.pi)
        angle = commanded_angle

        x = 0.0
        y = 0.0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)

        self.change_object_position("my_robot", x, y, commanded_angle)
        
        self.unpause.call_async(Empty.Request())
        
        # Wait for a valid pose update after teleporting
        wait_start = time.time()
        pose_info = None
        max_wait = 5.0 if self.use_ground_truth else 2.0
        while pose_info is None and (time.time() - wait_start) < max_wait:
            rclpy.spin_once(self, timeout_sec=0.1)
            pose_info = self.get_current_pose()
            
        if pose_info is not None:
            self.odom_x, self.odom_y, angle = pose_info
        else:
            self.get_logger().warn(f"Reset: Failed to get pose after {max_wait}s, using commanded position.")
            self.odom_x = float(x)
            self.odom_y = float(y)
            angle = commanded_angle
        
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

        self.prev_action = np.array([0.0, 0.0, 0.0])
        self.prev_theta = theta

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
        left_indices = range(left_start, left_end) if left_end > left_start else []
        if left_indices:
            left_danger = np.min(scan_array[left_start:left_end])
        else:
            left_danger = self.max_distance
        
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
        """Returns minimum laser distance for reward calculation."""
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
                # Reciprocal penalty
                # As dist approaches phys_limit, (dist - phys_limit) -> 0, penalty -> -inf
                # We add small epsilon to denominator for stability
                penalty = -1.0 / max(dist - phys_limit, 1e-3)
                penalty = max(penalty, -2000.0)
                total_barrier += penalty
                
        return total_barrier

    def get_reward(self, target, collision, action, min_laser, old_distance, new_distance, theta):
        """
        CBF-inspired soft barrier reward with modern components:
        - Potential-based progress reward (PBRS)
        - Velocity toward goal reward
        - Adaptive barrier scaling with floor and weight compensation
        - Enhanced heading reward (alignment + improvement)
        - Smoothness, angular, existence penalties
        
        Returns:
            tuple: (total_reward, reward_info_dict)
        """
        # Terminal reward for reaching the goal
        if target:
            return 500.0, {'total': 500.0, 'goal': 500.0}

        # ----- 1. POTENTIAL-BASED PROGRESS REWARD -----
        phi_old = -old_distance  # potential = -distance
        phi_new = -new_distance
        # REBALANCED: Increased to 2500 to account for TIME_DELTA=0.1
        # (0.1m progress * 2500 = 250 reward, comparable to 500 velocity reward)
        R_progress_pbrs = (phi_new - phi_old) * 2500.0

        # ----- 2. VELOCITY TOWARD GOAL REWARD -----
        # Compute unit vector from robot to goal
        if new_distance > 1e-6:
            goal_dir_x = (self.goal_x - self.odom_x) / new_distance
            goal_dir_y = (self.goal_y - self.odom_y) / new_distance
        else:
            goal_dir_x = 0.0
            goal_dir_y = 0.0
        v_toward_goal = action[0] * goal_dir_x + action[1] * goal_dir_y
        # REBALANCED: Increased to 500 to make velocity competitive with barrier penalties
        R_velocity = v_toward_goal * 500.0

        # ----- 3. ADAPTIVE BARRIER SCALING & COMPENSATION -----
        # CRITICAL: Use full_scan (360 rays) not scan_data (20 rays) for correct angle geometry
        R_barrier = self.get_barrier_reward(self.full_scan)
        R_barrier = max(R_barrier, -2000.0)  # safety cap
        distance_factor = max(0.33, min(1.0, new_distance / 3.0))  # floor at 1/3
        R_barrier_adaptive = R_barrier * distance_factor
        # Compensation: reallocate saved barrier weight to progress and velocity
        compensation = 1.0 - distance_factor
        R_progress_pbrs *= (1.0 + compensation * 0.5)
        R_velocity *= (1.0 + compensation * 0.5)

        # ----- 4. ENHANCED HEADING REWARD -----
        R_heading = math.cos(theta) * 5.0
        if not hasattr(self, 'prev_theta'):
            self.prev_theta = theta
        theta_improvement = abs(self.prev_theta) - abs(theta)
        R_heading_enhanced = R_heading + theta_improvement * 10.0
        self.prev_theta = theta

        # ----- 5. SMOOTHNESS PENALTY -----
        action_arr = np.array(action)
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.array([0.0, 0.0, 0.0])
        action_change = np.linalg.norm(action_arr - self.prev_action)
        R_smoothness = -action_change
        self.prev_action = action_arr.copy()

        # ----- 6. ANGULAR PENALTY -----
        R_angular = -abs(action[2])

        # ----- 7. EXISTENCE REWARD -----
        R_existence = 0.1

        # ----- 8. WEIGHTED SUM -----
        total_reward = (
            0.30 * R_progress_pbrs +
            0.20 * R_velocity +
            0.20 * R_barrier_adaptive +
            0.15 * R_heading_enhanced +
            0.05 * (R_smoothness * 5.0) +
            0.05 * (R_angular * 3.0) +
            0.05 * (R_existence * 1.0)
        )
        
        # Build info dict for logging
        reward_info = {
            'total': total_reward,
            'progress': R_progress_pbrs,
            'velocity': R_velocity,
            'barrier': R_barrier_adaptive,
            'heading': R_heading_enhanced,
            'smoothness': R_smoothness,
            'angular': R_angular,
            'existence': R_existence,
            'min_laser': min_laser,
            'distance_to_goal': new_distance
        }
        
        return total_reward, reward_info