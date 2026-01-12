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
COLLISION_DIST = 0.50 
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
        # Store scan parameters for collision detection
        self.scan_angle_min = scan.angle_min
        self.scan_angle_increment = scan.angle_increment
        
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
        
        robot_state = [distance, theta, action[0], action[1], action[2]]
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
        
        robot_state = [distance, theta, 0.0, 0.0, 0.0] 
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
        # Detect collision using Robot Bounding Box (0.4m x 0.2m)
        # Half-dimensions with safety margin
        ROBOT_HALF_LEN = 0.2 + 0.05  # 0.25m
        ROBOT_HALF_WIDTH = 0.1 + 0.05 # 0.15m
        
        min_laser = self.max_distance
        
        # Ensure we have scan parameters
        if not hasattr(self, 'scan_angle_min') or not hasattr(self, 'scan_angle_increment'):
             # Fallback to simple distance check if no scan received yet
             min_laser = min(laser_data)
             if min_laser < 0.35: # Increased safety margin
                 return True, True, min_laser
             return False, False, min_laser

        for i, r in enumerate(laser_data):
            if r == float('inf') or r == float('nan'):
                continue
                
            min_laser = min(min_laser, r)
            
            if r < 0.05: # Ignore self-hits or noise
                continue
            
            # Calculate point in robot frame
            angle = self.scan_angle_min + (i * self.scan_angle_increment)
            
            # Wrap angle to [-pi, pi] (optional but good practice)
            # angle = (angle + np.pi) % (2 * np.pi) - np.pi
            
            px = r * np.cos(angle)
            py = r * np.sin(angle)
            
            if abs(px) < ROBOT_HALF_LEN and abs(py) < ROBOT_HALF_WIDTH:
                self.get_logger().info(f"Collision detected at x={px:.2f}, y={py:.2f} (r={r:.2f})")
                return True, True, min_laser

        return False, False, min_laser

    def get_reward(self, target, collision, action, min_laser, old_distance, new_distance, theta):
        if target:
            return 500.0
        elif collision:
            return -200.0
        else:
            distance_change = old_distance - new_distance
            
            # STRONG progress reward - this is the MAIN incentive
            R_progress = distance_change * 400.0 
            
            # REMOVE heading reward - it encourages spinning to face goal
            # R_heading = 0.0 
            
            # VERY HIGH angular penalty to STOP circling
            penalty_angular = abs(action[2]) * 10.0 
            
            r3 = lambda x: 1 - x if x < 1 else 0.0
            penalty_collision = r3(min_laser) * 100.0 
            
            # Time penalty to encourage speed
            R_time = -0.1

            return R_progress - penalty_angular - penalty_collision + R_time