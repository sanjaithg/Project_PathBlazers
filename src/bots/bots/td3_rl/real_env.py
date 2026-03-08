import math
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray

GOAL_REACHED_DIST = 0.4
COLLISION_DIST = 0.50

class RealEnv(Node):
    def __init__(self, environment_dim):
        super().__init__('real_env')
        
        # ========================================================
        # USER CONFIGURABLE HARDWARE PARAMETERS
        # ========================================================
        self.cmd_vel_topic = "/cmd_vel"     # Command velocity topic
        self.odom_topic = "/odom"           # Odometry topic
        self.scan_topic = "/scan"           # LiDAR scan topic
        self.max_linear_speed = 0.5         # Max speed forward/backward (m/s)
        self.max_angular_speed = 0.5        # Max rotation speed (rad/s)
        self.robot_half_length = 0.3        # Half the length of the physical robot (m)
        self.robot_half_width = 0.3         # Half the width of the physical robot (m)
        self.lidar_max_distance = 3.5       # Max reliable LiDAR range (m)
        # ========================================================

        self.environment_dim = environment_dim
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.goal_x = 1.0
        self.goal_y = 0.0

        self.max_distance = self.lidar_max_distance
        self.scan_data = np.ones(self.environment_dim) * self.max_distance
        self.full_scan = np.ones(360) * self.max_distance
        self.last_odom = None

        self.vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 1)
        self.goal_point_publisher = self.create_publisher(MarkerArray, "goal_point", 3)

        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)

    def scan_callback(self, scan):
        self.scan_angle_min = scan.angle_min
        self.scan_angle_increment = scan.angle_increment
        
        ranges = np.nan_to_num(scan.ranges, nan=self.max_distance, posinf=self.max_distance)
        self.full_scan = np.clip(ranges, 1e-6, self.max_distance)
        
        mod = max(1, len(scan.ranges) // self.environment_dim)
        self.scan_data = [
            min(self.max_distance, ranges[i]) 
            for i in range(0, len(scan.ranges), mod)
        ]
        self.scan_data = np.maximum(np.array(self.scan_data[:self.environment_dim]), 1e-6)

    def odom_callback(self, msg):
        self.last_odom = msg

    def step(self, action):
        action = [float(a) for a in action]
        old_distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0]) * self.max_linear_speed
        vel_cmd.linear.y = float(action[1]) * self.max_linear_speed
        vel_cmd.angular.z = float(action[2]) * self.max_angular_speed
        self.vel_pub.publish(vel_cmd)
        
        self.publish_markers()

        done, collision, min_laser = self.observe_collision(self.full_scan)
        
        if self.last_odom is None:
             self.odom_x = 0.0
             self.odom_y = 0.0
             angle = 0.0
             self.get_logger().warn('No odom data received yet!', throttle_duration_sec=2.0)
        else:
             self.odom_x = float(self.last_odom.pose.pose.position.x)
             self.odom_y = float(self.last_odom.pose.pose.position.y)
             q = self.last_odom.pose.pose.orientation
             r = R.from_quat([q.x, q.y, q.z, q.w])
             euler = r.as_euler('xyz', degrees=False)
             angle = round(euler[2], 4)

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
        done = done or target or collision
        
        robot_state = [distance, theta, action[0], action[1], action[2]]
        state = np.append(self.scan_data, robot_state) 

        reward = 0.0 
        
        return state, reward, done, target

    def reset_state_to_current(self):
        if self.last_odom is None:
            angle = 0.0
            self.odom_x = 0.0
            self.odom_y = 0.0
        else:
            self.odom_x = float(self.last_odom.pose.pose.position.x)
            self.odom_y = float(self.last_odom.pose.pose.position.y)
            q = self.last_odom.pose.pose.orientation
            r = R.from_quat([q.x, q.y, q.z, q.w])
            euler = r.as_euler('xyz', degrees=False)
            angle = round(euler[2], 4)

        distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        
        target_angle = math.atan2(skew_y, skew_x)
        theta = target_angle - angle
        if theta > np.pi:
            theta -= 2 * np.pi
        if theta < -np.pi:
            theta += 2 * np.pi
            
        robot_state = [distance, theta, 0.0, 0.0, 0.0] 
        state = np.append(self.scan_data, robot_state)
        return state

    def publish_markers(self):
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
        ROBOT_HALF_LEN = self.robot_half_length 
        ROBOT_HALF_WIDTH = self.robot_half_width 
        min_laser = self.max_distance
        
        if not hasattr(self, 'scan_angle_min') or not hasattr(self, 'scan_angle_increment'):
             min_laser = min(laser_data) if len(laser_data) > 0 else self.max_distance
             if min_laser < 0.30: 
                 return True, True, min_laser
             return False, False, min_laser

        for i, r in enumerate(laser_data):
            if r == float('inf') or r == float('nan'):
                continue
            min_laser = min(min_laser, r)
            if r < 0.05: 
                continue
            angle = self.scan_angle_min + (i * self.scan_angle_increment)
            
            px = r * np.cos(angle)
            py = r * np.sin(angle)
            
            if abs(px) < ROBOT_HALF_LEN and abs(py) < ROBOT_HALF_WIDTH:
                self.get_logger().warn(f"Collision detected at x={px:.2f}, y={py:.2f} (r={r:.2f})")
                return True, True, min_laser

        return False, False, min_laser
